-- Finding high-quality households meters
WITH top_households AS (
  SELECT LCLid
  FROM `lcl_forecasting.raw_energy_data`
  GROUP BY LCLid
  -- therehold of 24,943 rows ensures 95% data availability for this project
  HAVING COUNT(*) >= 24943  
  ORDER BY COUNT(*) DESC
  LIMIT 100
),

-- calculate the aggregate mean of selected households
aggregated_energy AS (
  SELECT 
    DateTime,
    AVG(SAFE_CAST(`KWH_hh _per half hour_ ` AS FLOAT64)) as mean_consumption
  FROM `lcl_forecasting.raw_energy_data`
  WHERE LCLid IN (SELECT LCLid FROM top_households)
  /* THIS LINE DELETES THE 2011 NULL ROWS */
  AND DateTime >= '2012-01-01 00:00:00'
  GROUP BY DateTime
)

-- join weather features onto the energy time-series
SELECT 
  e.DateTime,
  e.mean_consumption,
  SAFE_CAST(w.temp AS FLOAT64) as temp,
  SAFE_CAST(w.humidity AS FLOAT64) as humidity
FROM aggregated_energy e
LEFT JOIN `lcl_forecasting.weather_hourly` w
  ON FORMAT_TIMESTAMP('%Y-%m-%d %H', e.DateTime) = FORMAT_TIMESTAMP('%Y-%m-%d %H', w.datetime)
ORDER BY e.DateTime ASC