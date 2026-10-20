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

var karaPikselleri = [
  [17030, 'SAMSUN BOLGE', 41.34583333335107,  36.24776558746591],
  [17033, 'ORDU',         40.97916666668424,  37.888919012799406],
  [17034, 'GIRESUN',      40.912500000017545, 38.39101854395677],
  [17112, 'CANAKKALE',    40.13750000001722,  26.406159580822123],
  [17119, 'YALOVA',       40.6541666666841,   29.27876107642282],
  [17310, 'ALANYA',       36.55416666668235,  31.97770026614934],
  [17340, 'MERSIN',       36.787500000015775, 34.603338234721384],
];

function noktaFC(liste, tur) {
  return ee.FeatureCollection(liste.map(function(r) {
    return ee.Feature(ee.Geometry.Point([r[3], r[2]]),
                      {istasyon_no: r[0], istasyon_ad: r[1], nokta: tur});
  }));
}
var noktalar = noktaFC(istasyonlar, 'istasyon').merge(noktaFC(karaPikselleri, 'kara_komsu'));
print('Nokta sayisi (82 istasyon + 7 kara):', noktalar.size());

var pNdvi = ee.Image(ee.ImageCollection('MODIS/061/MOD13A2').filterDate('2015-01-01', '2015-02-01').first())
              .select('NDVI').projection();
var pLst  = ee.Image(ee.ImageCollection('MODIS/061/MOD11A1').filterDate('2015-01-01', '2015-01-02').first())
              .select('LST_Day_1km').projection();
print('MOD13A2 NDVI crs:', pNdvi.crs(), 'transform:', pNdvi.transform(), 'nominal olcek (m):', pNdvi.nominalScale());
print('MOD11A1 LST  crs:', pLst.crs(),  'transform:', pLst.transform(),  'nominal olcek (m):', pLst.nominalScale());

// Olcek GEE katalogundan: NDVI x0.0001, gecerli aralik -2000..10000.
function ndviIsle(image) {
  var ham  = image.select('NDVI');
  var ndvi = ham.multiply(0.0001).updateMask(ham.gte(-2000)).rename('NDVI');
  var doy  = image.select('DayOfYear').toInt16().unmask(-1).rename('DayOfYear');
  var sqa  = image.select('SummaryQA').toInt16().unmask(-1).rename('SummaryQA');
  var dqa  = image.select('DetailedQA').toInt32().unmask(-1).rename('DetailedQA');
  return ndvi.addBands([doy, sqa, dqa])
             .set('tarih', image.date().format('YYYY-MM-dd'));
}
var ornek = ndviIsle(ee.Image(ee.ImageCollection('MODIS/061/MOD13A2').filterDate('2015-01-01', '2015-02-01').first()));
var pOrnek = ornek.select('NDVI').projection();
print('Islenmis NDVI crs:', pOrnek.crs(), 'transform:', pOrnek.transform());

// scale ve crs verilmiyor: ornekleme goruntunun kendi izgarasinda, noktayi iceren piksel okunur.
var YILLAR = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025];
var SUTUNLAR = ['istasyon_no', 'istasyon_ad', 'nokta', 'tarih',
                'NDVI', 'DayOfYear', 'SummaryQA', 'DetailedQA'];

YILLAR.forEach(function(yil) {
  var col = ee.ImageCollection('MODIS/061/MOD13A2')
    .filterDate(ee.Date(yil + '-01-01'), ee.Date((yil + 1) + '-01-01'))
    .filterBounds(noktalar.geometry().bounds())
    .select(['NDVI', 'DayOfYear', 'SummaryQA', 'DetailedQA'])
    .map(ndviIsle);

  var sonuc = col.map(function(image) {
    return image.reduceRegions({
      collection: noktalar,
      reducer   : ee.Reducer.first(),
      tileScale : 4
    }).map(function(f) {
      return f.set('tarih', image.get('tarih'));
    });
  }).flatten();

  Export.table.toDrive({
    collection    : sonuc,
    description   : 'MODIS_NDVI_YP_' + yil,
    folder        : 'MODIS_Turkiye',
    fileNamePrefix: 'MODIS_NDVI_YP_' + yil,
    fileFormat    : 'CSV',
    selectors     : SUTUNLAR
  });
});

Map.addLayer(noktalar.filter(ee.Filter.eq('nokta', 'istasyon')), {color: 'red'}, 'Istasyon');
Map.addLayer(noktalar.filter(ee.Filter.eq('nokta', 'kara_komsu')), {color: 'yellow'}, 'Kara pikseli');
Map.centerObject(noktalar, 6);

print('11 task olusturuldu. Tasks sekmesinden Run edin.');
