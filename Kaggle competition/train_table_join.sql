SET SEARCH_PATH to ferry_data;

--create join tables for train

CREATE TABLE traffic_work_hour as
(SELECT hour, year_month_day, avg(traffic_ordinal) as avg_traffic_ordinal
 FROM traffic_work
 GROUP BY  hour, year_month_day
);

CREATE TABLE horseshoe_join as
(SELECT vessel_name, full_date, scheduled_departure, avg_traffic_ordinal, avg_temperature, avg_dew_point, avg_Relative_Humidity_in_Percent, delay_indicator
 FROM train_work LEFT JOIN vancouver_work ON full_date = DATE(date_time) and EXTRACT(hour FROM scheduled_departure) = vancouver_work.hour
                             LEFT JOIN traffic_work_hour ON full_date = year_month_day and EXTRACT(hour FROM scheduled_departure) = traffic_work_hour.hour
 WHERE start_location = 'Horseshoe'
);

CREATE TABLE vancouver_join as
(SELECT vessel_name, full_date, scheduled_departure, avg_temperature, avg_dew_point, avg_Relative_Humidity_in_Percent, delay_indicator
 FROM train_work LEFT JOIN vancouver_work ON full_date = DATE(date_time) and EXTRACT(hour FROM scheduled_departure) = vancouver_work.hour
 WHERE start_location = 'vancouver'
);

CREATE TABLE victora_join as
(SELECT vessel_name, full_date, scheduled_departure, avg_temperature, avg_dew_point_temperature, avg_relative_humidity,
             avg_wind_direction, avg_wind_speed, avg_Visibility, avg_station_pressure, delay_indicator
 FROM  train_work LEFT JOIN victoria_work ON full_date = DATE(date_time) and EXTRACT(hour FROM scheduled_departure) = victoria_work.hour
 WHERE start_location = 'victoria'
);




--create join data for test

CREATE TABLE traffic_test_hour as
(SELECT hour, year_month_day, avg(traffic_ordinal) as avg_traffic_ordinal
 FROM traffic_test
 GROUP BY  hour, year_month_day
);

CREATE TABLE horseshoe_test_join as
(SELECT ID, vessel_name, full_date, scheduled_departure, avg_traffic_ordinal, avg_temperature, avg_dew_point, avg_Relative_Humidity_in_Percent
 FROM test LEFT JOIN vancouver_test ON full_date = DATE(date_time) and EXTRACT(hour FROM scheduled_departure) = vancouver_test.hour
                  LEFT JOIN traffic_test_hour ON full_date = year_month_day and EXTRACT(hour FROM scheduled_departure) = traffic_test_hour.hour
 WHERE start_location = 'Horseshoe'
);

CREATE TABLE vancouver_test_join as
(SELECT ID, vessel_name, full_date, scheduled_departure, avg_temperature, avg_dew_point, avg_Relative_Humidity_in_Percent
 FROM test LEFT JOIN vancouver_test ON full_date = DATE(date_time) and EXTRACT(hour FROM scheduled_departure) = vancouver_test.hour
 WHERE start_location = 'vancouver'
);

CREATE TABLE victora_test_join as
(SELECT ID, vessel_name, full_date, scheduled_departure, avg_temperature, avg_dew_point_temperature, avg_relative_humidity,
             avg_wind_direction, avg_wind_speed, avg_Visibility, avg_station_pressure
 FROM  test LEFT JOIN victoria_test ON full_date = DATE(date_time) and EXTRACT(hour FROM scheduled_departure) = victoria_test.hour
 WHERE start_location = 'victoria'
);