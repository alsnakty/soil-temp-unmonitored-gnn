var KARA_ESIK = 0.9;    // aday pikselde LST uretilen gun orani (2020) en az
var SU_ESIK   = 0.05;   // aday pikselin kapladigi alanda WorldCover su orani en fazla
var ARAMA_M   = 5000;   // arama yaricapi (m)
var KLASOR    = 'MODIS_Turkiye';

var yeni5 = [
  [17040, 'RIZE', 41.04, 40.5013],
  [17320, 'ANAMUR', 36.0686, 32.8649],
  [17026, 'SINOP', 42.0299, 35.1545],
  [17056, 'TEKIRDAG', 40.9585, 27.4965],
  [17302, 'ANTALYA BOLGE', 36.8851, 30.6828]
];
var mevcutKara = [
  [17030, 'SAMSUN BOLGE', 41.34583333335107, 36.24776558746591],
  [17033, 'ORDU', 40.97916666668424, 37.888919012799406],
  [17034, 'GIRESUN', 40.912500000017545, 38.39101854395677],
  [17112, 'CANAKKALE', 40.13750000001722, 26.406159580822123],
  [17119, 'YALOVA', 40.6541666666841, 29.27876107642282],
  [17310, 'ALANYA', 36.55416666668235, 31.97770026614934],
  [17340, 'MERSIN', 36.787500000015775, 34.603338234721384]
];

function fc(liste, tur) {
  return ee.FeatureCollection(liste.map(function(r) {
    return ee.Feature(ee.Geometry.Point([r[3], r[2]]),
      {istasyon_no: r[0], istasyon_ad: r[1], nokta_lat: r[2], nokta_lon: r[3], nokta_turu: tur});
  }));
}
var yeniFC = fc(yeni5, 'istasyon_yeni5');
var mevcutFC = fc(mevcutKara, 'kara_mevcut7');

var YARI_PIKSEL = 463;  // m, MODIS ~926 m pikselin yarisi (daire, pikselin ic bolgesine yakin)
var modis = ee.ImageCollection('MODIS/061/MOD11A1').filterBounds(yeniFC.merge(mevcutFC).geometry().bounds());
var modisProj = ee.Image(modis.filterDate('2020-01-01', '2020-01-02').first()).select('LST_Day_1km').projection();
var oran = modis.filterDate('2020-01-01', '2021-01-01').map(function(im) {
  return im.select('QC_Day').bitwiseAnd(3).neq(3).rename('oran');
}).mean().reproject(modisProj);
// WorldCover 80 = kalici su; kapsam disi (acik deniz) maskeli -> su sayilir.
var suWC = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map').eq(80).unmask(1).rename('su_piksel');
print('MODIS projeksiyon olcegi (m):', modisProj.nominalScale());

function suOrani(fc) {
  // fc noktalari -> YARI_PIKSEL cevresinde su orani; geometri nokta olarak geri konur
  return suWC.reduceRegions({
    collection: fc.map(function(f) { return f.set('nokta_geo', f.geometry()).setGeometry(f.geometry().buffer(YARI_PIKSEL)); }),
    reducer: ee.Reducer.mean().setOutputs(['su_piksel']), scale: 10, tileScale: 4
  }).map(function(f) { return f.setGeometry(ee.Geometry(f.get('nokta_geo'))); });
}

var tani = suOrani(oran.reduceRegions({collection: yeniFC.merge(mevcutFC), reducer: ee.Reducer.first().setOutputs(['oran']), tileScale: 4}));

function adaylar(f) {
  var p = f.geometry();
  var aday = oran.sample({region: p.buffer(ARAMA_M), projection: modisProj, geometries: true})
    .filter(ee.Filter.gte('oran', KARA_ESIK))
    .map(function(a) {
      var k = a.geometry().transform('EPSG:4326', 1).coordinates();
      return a.set({mesafe_m: a.geometry().distance(p, 1), kara_lon: k.get(0), kara_lat: k.get(1)});
    });
  return suOrani(aday).filter(ee.Filter.lte('su_piksel', SU_ESIK)).sort('mesafe_m');
}
// en yakin aday: limit(1) + aggregate_min (bos aday kumesinde hata vermeyen kalip, modis_lst_viewtime_qc_kiyi.js)
var secimYeni = tani.filter(ee.Filter.eq('nokta_turu', 'istasyon_yeni5')).map(function(f) {
  var a = adaylar(f);
  var en = a.limit(1);
  return f.set({kara_lat: en.aggregate_min('kara_lat'), kara_lon: en.aggregate_min('kara_lon'),
                mesafe_m: en.aggregate_min('mesafe_m'), kara_oran: en.aggregate_min('oran'),
                kara_su: en.aggregate_min('su_piksel'), aday_sayisi: a.size()});
});
var secim = secimYeni.merge(tani.filter(ee.Filter.eq('nokta_turu', 'kara_mevcut7')));
Export.table.toDrive({
  collection: secim, description: 'kiyi5_kara_pikseli', folder: KLASOR,
  fileNamePrefix: 'kiyi5_kara_pikseli', fileFormat: 'CSV',
  selectors: ['nokta_turu', 'istasyon_no', 'istasyon_ad', 'nokta_lat', 'nokta_lon', 'oran', 'su_piksel',
              'kara_lat', 'kara_lon', 'mesafe_m', 'kara_oran', 'kara_su', 'aday_sayisi']
});

var karaFC = secimYeni
  .filter(ee.Filter.notNull(['kara_lat']))
  .map(function(f) {
    return ee.Feature(ee.Geometry.Point([f.get('kara_lon'), f.get('kara_lat')]),
      {istasyon_no: f.get('istasyon_no'), istasyon_ad: f.get('istasyon_ad'), nokta: 'kara_komsu'});
  });
print('Kara pikseli bulunan YENI5 istasyonu (5 olmali):', karaFC.size());

function lstIsle(image) {
  var lstDay = image.select('LST_Day_1km'), lstNight = image.select('LST_Night_1km');
  return lstDay.multiply(0.02).subtract(273.15).updateMask(lstDay.gt(0)).rename('LST_Day_C')
    .addBands([
      lstNight.multiply(0.02).subtract(273.15).updateMask(lstNight.gt(0)).rename('LST_Night_C'),
      image.select('QC_Day').toInt16().unmask(-1).rename('QC_Day'),
      image.select('QC_Night').toInt16().unmask(-1).rename('QC_Night'),
      image.select('Day_view_time').multiply(0.1).rename('Day_view_time'),
      image.select('Night_view_time').multiply(0.1).rename('Night_view_time'),
      image.select('Day_view_angle').subtract(65).rename('Day_view_angle'),
      image.select('Night_view_angle').subtract(65).rename('Night_view_angle')])
    .set('tarih', image.date().format('YYYY-MM-dd'));
}
var lstSeri = ee.ImageCollection('MODIS/061/MOD11A1').filterDate('2015-01-01', '2026-01-01')
  .filterBounds(karaFC.geometry().bounds())
  .select(['LST_Day_1km', 'LST_Night_1km', 'QC_Day', 'QC_Night', 'Day_view_time', 'Night_view_time',
           'Day_view_angle', 'Night_view_angle'])
  .map(lstIsle)
  .map(function(im) {
    return im.reduceRegions({collection: karaFC, reducer: ee.Reducer.first(), tileScale: 4})
      .map(function(f) { return f.set('tarih', im.get('tarih')); });
  }).flatten();
Export.table.toDrive({
  collection: lstSeri, description: 'MODIS_LST_KIYI5', folder: KLASOR, fileNamePrefix: 'MODIS_LST_KIYI5',
  fileFormat: 'CSV',
  selectors: ['istasyon_no', 'istasyon_ad', 'nokta', 'tarih', 'LST_Day_C', 'LST_Night_C', 'QC_Day', 'QC_Night',
              'Day_view_time', 'Night_view_time', 'Day_view_angle', 'Night_view_angle']
});

function ndviIsle(image) {
  var ham = image.select('NDVI');
  return ham.multiply(0.0001).updateMask(ham.gte(-2000)).rename('NDVI')
    .addBands([image.select('DayOfYear').toInt16().unmask(-1).rename('DayOfYear'),
               image.select('SummaryQA').toInt16().unmask(-1).rename('SummaryQA'),
               image.select('DetailedQA').toInt32().unmask(-1).rename('DetailedQA')])
    .set('tarih', image.date().format('YYYY-MM-dd'));
}
var ndviSeri = ee.ImageCollection('MODIS/061/MOD13A2').filterDate('2015-01-01', '2026-01-01')
  .filterBounds(karaFC.geometry().bounds())
  .select(['NDVI', 'DayOfYear', 'SummaryQA', 'DetailedQA'])
  .map(ndviIsle)
  .map(function(im) {
    return im.reduceRegions({collection: karaFC, reducer: ee.Reducer.first(), tileScale: 4})
      .map(function(f) { return f.set('tarih', im.get('tarih')); });
  }).flatten();
Export.table.toDrive({
  collection: ndviSeri, description: 'MODIS_NDVI_KIYI5', folder: KLASOR, fileNamePrefix: 'MODIS_NDVI_KIYI5',
  fileFormat: 'CSV',
  selectors: ['istasyon_no', 'istasyon_ad', 'nokta', 'tarih', 'NDVI', 'DayOfYear', 'SummaryQA', 'DetailedQA']
});

Map.addLayer(yeniFC, {color: 'red'}, 'Istasyon (yeni5)');
Map.addLayer(karaFC, {color: 'yellow'}, 'Secilen kara pikseli');
Map.addLayer(mevcutFC, {color: 'orange'}, 'Mevcut 7 kara pikseli');
Map.centerObject(yeniFC, 6);
print('3 task olusturuldu (kiyi5_kara_pikseli, MODIS_LST_KIYI5, MODIS_NDVI_KIYI5). Tasks sekmesinden Run edin.');
