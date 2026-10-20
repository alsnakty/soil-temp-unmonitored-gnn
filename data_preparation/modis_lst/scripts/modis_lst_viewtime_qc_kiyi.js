var istasyonlar = [
  [17020, 'BARTIN', 41.6248, 32.3569],
  [17022, 'ZONGULDAK', 41.449242, 31.777917],
  [17026, 'SINOP', 42.0299, 35.1545],
  [17030, 'SAMSUN BOLGE', 41.344167, 36.256389],
  [17033, 'ORDU', 40.9838, 37.8858],
  [17034, 'GIRESUN', 40.9227, 38.3878],
  [17037, 'TRABZON BOLGE', 40.9985, 39.7649],
  [17040, 'RIZE', 41.04, 40.5013],
  [17045, 'ARTVIN', 41.1752, 41.8187],
  [17046, 'ARDAHAN', 41.1061, 42.7055],
  [17050, 'EDIRNE', 41.6767, 26.5508],
  [17052, 'KIRKLARELI', 41.7382, 27.2178],
  [17056, 'TEKIRDAG', 40.9585, 27.4965],
  [17064, 'ISTANBUL BOLGE', 40.9113, 29.1558],
  [17066, 'KOCAELI', 40.7663, 29.9173],
  [17069, 'SAKARYA', 40.7676, 30.3934],
  [17070, 'BOLU', 40.7329, 31.6022],
  [17072, 'DUZCE', 40.8437, 31.1488],
  [17074, 'KASTAMONU', 41.371, 33.7756],
  [17077, 'KARABUK KAPULLU', 41.2327, 32.6294],
  [17080, 'CANKIRI', 40.6082, 33.6102],
  [17084, 'CORUM', 40.5461, 34.9362],
  [17085, 'AMASYA', 40.6668, 35.8353],
  [17086, 'TOKAT', 40.3312, 36.5577],
  [17088, 'GUMUSHANE', 40.4598, 39.4653],
  [17089, 'BAYBURT', 40.2547, 40.2207],
  [17090, 'SIVAS', 39.7437, 37.002],
  [17094, 'ERZINCAN', 39.7523, 39.4868],
  [17095, 'ERZURUM BOLGE', 39.9058, 41.2544],
  [17097, 'KARS', 40.6042, 43.1073],
  [17099, 'AGRI', 39.7253, 43.0522],
  [17100, 'IGDIR', 39.9227, 44.0523],
  [17112, 'CANAKKALE', 40.141, 26.3993],
  [17116, 'BURSA', 40.2308, 29.0133],
  [17119, 'YALOVA', 40.6589, 29.2796],
  [17120, 'BILECIK', 40.1414, 29.9772],
  [17126, 'ESKISEHIR BOLGE', 39.7656, 30.5502],
  [17130, 'ANKARA BOLGE', 39.9727, 32.8637],
  [17135, 'KIRIKKALE', 39.8433, 33.5181],
  [17140, 'YOZGAT', 39.8243, 34.8159],
  [17155, 'KUTAHYA', 39.4171, 29.9891],
  [17160, 'KIRSEHIR', 39.1639, 34.1561],
  [17165, 'TUNCELI', 39.1058, 39.5408],
  [17172, 'VAN BOLGE', 38.4693, 43.346],
  [17186, 'MANISA', 38.6153, 27.4049],
  [17188, 'USAK', 38.671233, 29.40404],
  [17190, 'AFYONKARAHISAR BOLGE', 38.738, 30.5604],
  [17192, 'AKSARAY', 38.3705, 33.9987],
  [17193, 'NEVSEHIR', 38.6163, 34.7025],
  [17196, 'KAYSERI BOLGE', 38.687, 35.5],
  [17199, 'MALATYA', 38.3367, 38.2173],
  [17201, 'ELAZIG BOLGE', 38.6443, 39.2561],
  [17203, 'BINGOL', 38.8847, 40.5007],
  [17204, 'MUS', 38.7509, 41.5023],
  [17208, 'BITLIS', 38.475, 42.1625],
  [17210, 'SIIRT', 37.9319, 41.9354],
  [17220, 'IZMIR BOLGE', 38.3949, 27.0819],
  [17234, 'AYDIN', 37.8402, 27.8379],
  [17237, 'DENIZLI', 37.762, 29.0921],
  [17238, 'BURDUR', 37.722, 30.294],
  [17240, 'ISPARTA', 37.7848, 30.5679],
  [17245, 'KONYA BOLGE', 37.8687, 32.4713],
  [17246, 'KARAMAN', 37.1932, 33.2202],
  [17250, 'NIGDE', 37.9587, 34.6795],
  [17255, 'KAHRAMANMARAS', 37.576, 36.915],
  [17261, 'GAZIANTEP', 37.0585, 37.351],
  [17262, 'KILIS', 36.7085, 37.1123],
  [17265, 'ADIYAMAN', 37.7553, 38.2775],
  [17270, 'SANLIURFA', 37.1608, 38.7863],
  [17275, 'MARDIN', 37.3103, 40.7284],
  [17282, 'BATMAN', 37.8636, 41.1562],
  [17285, 'HAKKARI', 37.5745, 43.7388],
  [17287, 'SIRNAK', 37.5209, 42.4523],
  [17292, 'MUGLA', 37.2095, 28.3668],
  [17302, 'ANTALYA BOLGE', 36.8851, 30.6828],
  [17310, 'ALANYA', 36.5499, 31.9798],
  [17320, 'ANAMUR', 36.0686, 32.8649],
  [17340, 'MERSIN', 36.7808, 34.6031],
  [17351, 'ADANA BOLGE', 37.0041, 35.3443],
  [17355, 'OSMANIYE', 37.1021, 36.2539],
  [17372, 'ANTAKYA', 36.2048, 36.1513],
  [17636, 'FLORYA', 40.9758, 28.7865],
];

var BAS = '2015-01-01';
var BIT = '2026-01-01';
var ORAN_BAS = '2020-01-01';   // LST uretilen gun orani bu yil uzerinden
var ORAN_BIT = '2021-01-01';
var YILLAR = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025];
var KIYI_ESIK  = 0.1;    // istasyon pikselinde LST uretilen gun orani bunun altindaysa kiyi sayilir
var KARA_ESIK  = 0.9;    // aday kara pikselinde LST uretilen gun orani en az bu olmali
var ARAMA_M    = 5000;   // en yakin kara pikseli arama yaricapi (m)
var SU_YARICAP = 500;    // WorldCover su orani yaricapi (m), 1 km MODIS pikselinin yaklasigi
var KLASOR = 'MODIS_Turkiye';

var istFC = ee.FeatureCollection(
  istasyonlar.map(function(ist) {
    return ee.Feature(
      ee.Geometry.Point([ist[3], ist[2]]),
      {istasyon_no: ist[0], istasyon_ad: ist[1], nokta: 'istasyon',
       ist_lat: ist[2], ist_lon: ist[3]}
    );
  })
);
print('Istasyon sayisi:', istFC.size());

var BANTLAR = ['LST_Day_1km', 'LST_Night_1km', 'QC_Day', 'QC_Night',
               'Day_view_time', 'Night_view_time', 'Day_view_angle', 'Night_view_angle'];

var modisTum = ee.ImageCollection('MODIS/061/MOD11A1')
  .filterDate(BAS, BIT)
  .filterBounds(istFC.geometry().bounds())
  .select(BANTLAR);

var modisProj = ee.Image(modisTum.first()).select('LST_Day_1km').projection();

// QC_Day bit 0-1: 0 iyi, 1 diger kalite, 2 bulut nedeniyle uretilmedi,
// 3 bulut disi nedenle uretilmedi.
function lstIsle(image) {
  var lstDay   = image.select('LST_Day_1km');
  var lstNight = image.select('LST_Night_1km');

  var dayC   = lstDay.multiply(0.02).subtract(273.15)
                 .updateMask(lstDay.gt(0)).rename('LST_Day_C');
  var nightC = lstNight.multiply(0.02).subtract(273.15)
                 .updateMask(lstNight.gt(0)).rename('LST_Night_C');
  var qcDay   = image.select('QC_Day').toInt16().unmask(-1).rename('QC_Day');
  var qcNight = image.select('QC_Night').toInt16().unmask(-1).rename('QC_Night');
  var vtDay   = image.select('Day_view_time').multiply(0.1).rename('Day_view_time');
  var vtNight = image.select('Night_view_time').multiply(0.1).rename('Night_view_time');
  var vaDay   = image.select('Day_view_angle').subtract(65).rename('Day_view_angle');
  var vaNight = image.select('Night_view_angle').subtract(65).rename('Night_view_angle');

  return dayC
    .addBands([nightC, qcDay, qcNight, vtDay, vtNight, vaDay, vaNight])
    .set('tarih', image.date().format('YYYY-MM-dd'));
}

var oranTum = modisTum.filterDate(ORAN_BAS, ORAN_BIT).map(function(image) {
  return image.select('QC_Day').bitwiseAnd(3).neq(3).rename('oran');
}).mean();
var gecerliOran = oranTum.reproject(modisProj);

var tani1000 = oranTum.reduceRegions({
  collection: istFC,
  reducer   : ee.Reducer.first().setOutputs(['oran_1000']),
  crs       : modisProj,
  scale     : 1000,
  tileScale : 4
});
var tani = gecerliOran.reduceRegions({
  collection: tani1000,
  reducer   : ee.Reducer.first().setOutputs(['oran_yerel']),
  tileScale : 4
});

// WorldCover sinif 80 = kalici su. Kapsam disi (acik deniz) maskeli geldigi icin su sayilir.
var su = ee.ImageCollection('ESA/WorldCover/v200').first()
  .select('Map').eq(80).unmask(1).rename('su');

var taniTablo = su.reduceRegions({
  collection: tani.map(function(f) { return f.buffer(SU_YARICAP); }),
  reducer   : ee.Reducer.mean().setOutputs(['su_orani_500m']),
  scale     : 10,
  tileScale : 4
});

Export.table.toDrive({
  collection    : taniTablo,
  description   : 'LST_tani_istasyon',
  folder        : KLASOR,
  fileNamePrefix: 'LST_tani_istasyon',
  fileFormat    : 'CSV',
  selectors     : ['istasyon_no', 'istasyon_ad', 'ist_lat', 'ist_lon',
                   'oran_1000', 'oran_yerel', 'su_orani_500m']
});

var kiyiFC = tani.filter(ee.Filter.lt('oran_yerel', KIYI_ESIK));

function adaylarBul(f) {
  var nokta = f.geometry();
  return gecerliOran.rename('oran').sample({
    region    : nokta.buffer(ARAMA_M),
    projection: modisProj,
    geometries: true
  }).map(function(a) {
    var k = a.geometry().transform('EPSG:4326', 1).coordinates();
    return a.set({mesafe_m: a.geometry().distance(nokta, 1),
                  kara_lon: k.get(0), kara_lat: k.get(1)});
  });
}

// Tani satiri her kiyi istasyonu icin yazilir; aday yoksa kara_* bos kalir.
var kiyiTablo = kiyiFC.map(function(f) {
  var tum = adaylarBul(f);
  var gecerli = tum.filter(ee.Filter.gte('oran', KARA_ESIK));
  var enYakin = gecerli.sort('mesafe_m').limit(1);
  return f.set({
    piksel_sayisi_5km: tum.size(),
    aday_sayisi      : gecerli.size(),
    en_iyi_oran_5km  : tum.aggregate_max('oran'),
    mesafe_m         : enYakin.aggregate_min('mesafe_m'),
    kara_lat         : enYakin.aggregate_min('kara_lat'),
    kara_lon         : enYakin.aggregate_min('kara_lon'),
    kara_gecerli_oran: enYakin.aggregate_min('oran')
  });
});

Export.table.toDrive({
  collection    : kiyiTablo,
  description   : 'LST_kiyi_kara_pikseli',
  folder        : KLASOR,
  fileNamePrefix: 'LST_kiyi_kara_pikseli',
  fileFormat    : 'CSV',
  selectors     : ['istasyon_no', 'istasyon_ad', 'ist_lat', 'ist_lon', 'oran_1000', 'oran_yerel',
                   'piksel_sayisi_5km', 'aday_sayisi', 'en_iyi_oran_5km',
                   'kara_lat', 'kara_lon', 'mesafe_m', 'kara_gecerli_oran']
});

// Zaman serisinde kullanilacak kara noktalari (aday bulunamayan istasyon dusmez,
// yalniz kara_komsu satiri olmaz).
var karaFC = kiyiFC.map(function(f) {
  return adaylarBul(f)
    .filter(ee.Filter.gte('oran', KARA_ESIK))
    .sort('mesafe_m').limit(1)
    .map(function(a) {
      return ee.Feature(a.geometry(), {
        istasyon_no: f.get('istasyon_no'),
        istasyon_ad: f.get('istasyon_ad'),
        nokta      : 'kara_komsu'
      });
    });
}).flatten();

Map.addLayer(gecerliOran, {min: 0, max: 1, palette: ['0000ff', 'ffffff', '00aa00']},
             'LST uretilen gun orani (2020)', false);
Map.addLayer(istFC, {color: 'red'}, '82 Istasyon');
Map.centerObject(istFC, 6);

var SUTUNLAR = ['istasyon_no', 'istasyon_ad', 'nokta', 'tarih',
                'LST_Day_C', 'LST_Night_C', 'QC_Day', 'QC_Night',
                'Day_view_time', 'Night_view_time', 'Day_view_angle', 'Night_view_angle'];

YILLAR.forEach(function(yil) {
  var col = modisTum
    .filterDate(ee.Date(yil + '-01-01'), ee.Date((yil + 1) + '-01-01'))
    .map(lstIsle);

  var sonuc = col.map(function(image) {
    var ist = image.reduceRegions({
      collection: istFC,
      reducer   : ee.Reducer.first(),
      scale     : 1000,
      tileScale : 4
    });
    var kara = image.reduceRegions({
      collection: karaFC,
      reducer   : ee.Reducer.first(),
      tileScale : 4
    });
    return ee.FeatureCollection([ist, kara]).flatten().map(function(f) {
      return f.set('tarih', image.get('tarih'));
    });
  }).flatten();

  Export.table.toDrive({
    collection    : sonuc,
    description   : 'MODIS_LST_VT_' + yil,
    folder        : KLASOR,
    fileNamePrefix: 'MODIS_LST_VT_' + yil,
    fileFormat    : 'CSV',
    selectors     : SUTUNLAR
  });
});

print('13 task olusturuldu (2 tani + 11 yil). Once 2 tani task, sonra yillar.');
