 DROP SCHEMA IF EXISTS ferry_data CASCADE;
CREATE SCHEMA ferry_data;

SET SEARCH_PATH to ferry_data;

CREATE TABLE train_work(
  vessel_name VARCHAR(50),
  scheduled_departure TIME NOT NULL,
  status VARCHAR(100),
  trip VARCHAR(50),
  trip_Duration INT,
  day VARCHAR(20),
  month VARCHAR(20),
  day_of_month INT,
  year INT,
  full_date DATE NOT NULL,
  delay_indicator INT NOT NULL,
PRIMARY KEY(vessel_name, full_date, scheduled_departure)
);

CREATE TABLE traffic_work(
  year INT,
  month INT,
  day INT,
  hour INT,
  minute INT,
  second INT,
  traffic_ordinal INT,
  year_month_day DATE NOT NULL,
PRIMARY KEY(year_month_day, hour, minute, second)
);

CREATE TABLE vancouver_work(
  date_time TIMESTAMP PRIMARY KEY,
  year INT,
  month INT,
  day INT,
  time TIME,
  temperature_in_celsius REAL,
  dew_point_temperature_in_celsius REAL,
  relative_humidity_in_percent INT,
  humidex_in_celsius REAL,
  hour INT
);

CREATE TABLE victoria_work(
  date_time TIMESTAMP PRIMARY KEY,
  year INT,
  month INT,
  day INT,
  time TIME,
  temperature_in_celsius REAL,
  dew_point_temperature_in_celsius REAL,
  relative_humidity_in_percent INT,
  wind_direction_in_degrees INT,
  wind_speed_km_per_h INT,
  Visibility_in_km REAL,
  station_pressure_in_kPa REAL,
  weather VARCHAR(30),
  hour INT
);


\COPY train_work FROM 'train_work.csv' DELIMITER ',' CSV header;
\COPY traffic_work FROM 'traffic_work.csv' DELIMITER ',' CSV header;
\COPY vancouver_work FROM 'vancouver_work.csv' DELIMITER ',' CSV header;
\COPY victoria_work FROM 'victoria_work.csv' DELIMITER ',' CSV header;

CREATE INDEX train_work_inx ON train_work(scheduled_departure);
CREATE INDEX traffic_work_inx ON traffic_work(year_month_day);
CREATE INDEX vancouver_work_inx ON vancouver_work(date_time);
CREATE INDEX victoria_work_inx ON victoria_work(date_time);
