import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, SAGEConv
from torch_geometric.utils import softmax as kenar_softmax

from ortak_ayarlar import Config, N_VAR

RELATIONS = ("V", "H", "T")
EDGE_DIM = {"V": 1, "H": 2, "T": 1}
UZAKLIK_MERKEZ_KM = tuple(range(0, 201, 25))
UZAKLIK_GENISLIK_KM = 25.0


class GatedEdgeConv(MessagePassing):
    """z_i = ReLU(W_self h_i + mean_j (W_neigh h_j) * sigmoid(W_edge e_ij))."""

    def __init__(self, dim, edge_dim):
        super().__init__(aggr="mean")
        self.lin_self = nn.Linear(dim, dim)
        self.lin_neigh = nn.Linear(dim, dim)
        self.lin_edge = nn.Linear(edge_dim, dim)

    def forward(self, x, edge_index, edge_attr):
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return F.relu(self.lin_self(x) + agg)

    def message(self, x_j, edge_attr):
        return self.lin_neigh(x_j) * torch.sigmoid(self.lin_edge(edge_attr))


class SoftmaxUzaklikConv(MessagePassing):

    def __init__(self, dim):
        super().__init__(aggr="add")
        self.lin_self = nn.Linear(dim, dim)
        self.lin_neigh = nn.Linear(dim, dim)
        self.skor = nn.Sequential(nn.Linear(len(UZAKLIK_MERKEZ_KM), 16), nn.ReLU(), nn.Linear(16, 1))
        self.register_buffer("mu", torch.tensor(UZAKLIK_MERKEZ_KM, dtype=torch.float32))

    def skor_d(self, d):
        """Kenar skoru, d (E, 1) km -> (E,). Duman testindeki kimlik sinamasi bunu -p ln(d) ile degistirir."""
        phi = torch.exp(-0.5 * ((d - self.mu[None, :]) / UZAKLIK_GENISLIK_KM) ** 2)
        return self.skor(phi).squeeze(-1)

    def agirlik(self, edge_index, edge_attr, kaynak_gorunur, n):
        sec = kaynak_gorunur[edge_index[0]]
        ei, d = edge_index[:, sec], edge_attr[sec, 0:1] * 1000.0
        a = kenar_softmax(self.skor_d(d), ei[1], num_nodes=n)
        return ei, a

    def forward(self, x, edge_index, edge_attr, kaynak_gorunur):
        ei, a = self.agirlik(edge_index, edge_attr, kaynak_gorunur, x.size(0))
        agg = self.propagate(ei, x=x, a=a)
        return F.relu(self.lin_self(x) + agg)

    def message(self, x_j, a):
        return a[:, None] * self.lin_neigh(x_j)


class GNNModel(nn.Module):
    """Ablasyonlar bayraklarla: drop=("V",)/("H",), fixed_fusion=True, flat=True."""

    def __init__(self, cfg: Config, drop=(), fixed_fusion=False, flat=False,
                 in_dim=N_VAR, target_cols=None, edge_dims=None, maske_cols=None, idw_artik=False):
        super().__init__()
        # idw_artik: batch.base (normalize IDW_rakim tahmini) ek girdi; cikti = base + duzeltme (tek hedef sutunu)
        # maske_cols: eksik-deger gostergesi istenen girdi sutunlari; her biri icin 0/1 (1 = NaN) ek girdi.
        self.maske_cols = list(maske_cols or [])
        self.target_cols = list(range(in_dim)) if target_cols is None else list(target_cols)
        self.idw_artik = bool(idw_artik)
        if self.idw_artik and len(self.target_cols) != 1:
            raise ValueError("idw_artik yalniz tek hedef sutunuyla.")
        edge_dims = EDGE_DIM if edge_dims is None else edge_dims
        d, q = cfg.hidden_dim, cfg.hidden_dim // 4
        if len(set(drop)) >= len(RELATIONS):
            raise ValueError("En az bir iliski tipi kalmali.")
        if flat and (drop or fixed_fusion):
            raise ValueError("flat modelde fuzyon/drop yok.")
        self.flat, self.fixed_fusion, self.L = flat, fixed_fusion, cfg.n_layers

        self.inp = nn.Linear(in_dim + len(self.maske_cols) + int(self.idw_artik), d)
        self.pe_p, self.pe_lat, self.pe_lon = nn.Linear(1, q), nn.Linear(1, q), nn.Linear(1, q)
        self.pe_hour = nn.Linear(2, q)
        self.pe_merge = nn.Linear(2 * d, d)
        self.dropout = nn.Dropout(cfg.dropout)

        if flat:
            self.convs = nn.ModuleList([SAGEConv(d, d) for _ in range(self.L)])
        else:
            self.yatay = getattr(cfg, "yatay", "kapi")
            if self.yatay not in ("kapi", "softmax"):
                raise ValueError(self.yatay)
            conv = lambda k: (SoftmaxUzaklikConv(d) if (k == "H" and self.yatay == "softmax")
                              else GatedEdgeConv(d, edge_dims[k]))
            self.convs = nn.ModuleList([nn.ModuleDict({k: conv(k) for k in RELATIONS}) for _ in range(self.L)])
            self.fusion_w = nn.ParameterList(
                [nn.Parameter(torch.zeros(len(RELATIONS))) for _ in range(self.L)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(self.L)])
        self.register_buffer("drop_mask", torch.tensor([k in drop for k in RELATIONS]))
        self.register_buffer("fixed_w", torch.full((len(RELATIONS),), 1.0 / len(RELATIONS)))

        self.attn = nn.MultiheadAttention(d, cfg.attn_heads, batch_first=True)
        self.out = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(cfg.dropout),
                                 nn.Linear(d, len(self.target_cols)))
        if self.idw_artik:   # baslangicta duzeltme 0: model egitim basinda IDW_rakim'in kendisi
            nn.init.zeros_(self.out[-1].weight)
            nn.init.zeros_(self.out[-1].bias)

    def effective_edge_weights(self, layer: int) -> torch.Tensor:
        if self.fixed_fusion:
            return self.fixed_w
        w = self.fusion_w[layer]
        if bool(self.drop_mask.any()):
            w = w.masked_fill(self.drop_mask, float("-inf"))
        return torch.softmax(w, dim=0)

    @torch.no_grad()
    def get_edge_weights(self):
        if self.flat:
            return []
        return [dict(zip(RELATIONS, self.effective_edge_weights(l).cpu().tolist()))
                for l in range(self.L)]

    def forward(self, batch):
        x = batch.x
        xt = x[:, self.target_cols]
        miss = torch.isnan(xt)
        x0 = torch.nan_to_num(x, nan=0.0)
        if self.maske_cols:
            x0 = torch.cat([x0, torch.isnan(x[:, self.maske_cols]).to(x0.dtype)], dim=-1)
        if self.idw_artik:
            if batch.base is None:
                raise ValueError("idw_artik acik ama batch.base yok (cfg.idw_artik kapali mi?).")
            taban = torch.nan_to_num(batch.base, nan=0.0)   # hesaplanamayan -> derinlik ortalamasi
            x0 = torch.cat([x0, taban], dim=-1)
        h = self.inp(x0)
        p = batch.pos
        pe = torch.cat([self.pe_p(p[:, 0:1]), self.pe_lat(p[:, 1:2]),
                        self.pe_lon(p[:, 2:3]), self.pe_hour(p[:, 3:5])], dim=-1)
        h = self.dropout(self.pe_merge(torch.cat([h, pe], dim=-1)))

        if self.flat:
            ei_all = torch.cat([batch.edge_index[k] for k in RELATIONS], dim=1)
        for l in range(self.L):
            if self.flat:
                z = F.relu(self.convs[l](h, ei_all))
            else:
                a = self.effective_edge_weights(l)
                z = 0
                for i, k in enumerate(RELATIONS):
                    if bool(self.drop_mask[i]):
                        continue
                    if k == "H" and self.yatay == "softmax":   # kaynak gorunur: hedef sutunu olculmus ve gizli degil
                        z = z + a[i] * self.convs[l][k](h, batch.edge_index[k], batch.edge_attr[k], ~miss[:, 0])
                    else:
                        z = z + a[i] * self.convs[l][k](h, batch.edge_index[k], batch.edge_attr[k])
            h = self.norms[l](h + z)

        B, W, S = batch.B, batch.W, batch.S
        assert B * W * S == h.size(0), "B*W*S != N"
        d = h.size(-1)
        hh = h.view(B, W, S, d).permute(0, 2, 1, 3).reshape(B * S, W, d)
        kpm = batch.time_pad[:, None, :].expand(B, S, W).reshape(B * S, W)
        att, _ = self.attn(hh, hh, hh, key_padding_mask=kpm if bool(kpm.any()) else None,
                           need_weights=False)
        hh = hh + att
        h = hh.view(B, S, W, d).permute(0, 2, 1, 3).reshape(B * W * S, d)

        out = self.out(h)
        if self.idw_artik:
            out = out + taban
        return torch.where(miss, out, xt)


def build_model(cfg: Config, opts: dict, **kw) -> GNNModel:
    return GNNModel(cfg, drop=opts["drop"], fixed_fusion=opts["fixed_fusion"], flat=opts["flat"], **kw)
