-- Dm Prediction table:
CREATE TABLE dm_prediction (
	gender VARCHAR(10) NOT NULL,
	age	DECIMAL NOT NULL,
	hypertension INTEGER NOT NULL, 
	heart_disease INTEGER NOT NULL, 	
	smoking_history	VARCHAR(15) NOT NULL,
	bmi	DECIMAL NOT NULL,
	HbA1c_level	DECIMAL NOT NULL,
	blood_glucose_level	INTEGER NOT NULL, 
	diabetes INTEGER NOT NULL
	);
	


-- View the dm_prediction table:
SELECT *
FROM dm_prediction
;