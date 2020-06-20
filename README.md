## Predicting Prediabetes with BigQuery ML using FHIR and SDOH data
This repo mainly contains a [jupyter notebook](/SDOH-ML.ipynb) that builds a [Binary Logistic Regression](https://towardsdatascience.com/implementing-binary-logistic-regression-in-r-7d802a9d98fe) model using BigQueryML to predict diabetes for a population of FHIR patient data that is augmented by SDOH factors such as food and transportation. 

## BigQuery ML and Binary Logistic Regression Model
You can use the binary logistic regression model to predict whether a value falls into one of two categories. A common problem in machine learning is to classify data into one of two types, known as labels. In this case, we may want to predict whether a given patient will be prediabetic or not , based on other information about that patient. The two labels will be "prediabetic" and "not prediabetic".  The input dataset needs to be built such that one column represents the label. 

[Google's BigQuery ML](https://cloud.google.com/bigquery-ml/docs/bigqueryml-intro) enables users to create and execute machine learning models in BigQuery using standard SQL queries. BigQuery ML supports supervised learning  with the [logistic regression](https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-create#model_type) model type.

## Patient Data - Synthea FHIR 
The primary dataset that's used in this exercise is from [Synthea Synthetic Patient FHIR data](https://synthetichealth.github.io/synthea/) which contains around ~1M Patient resources and their related Conditions and Observations. This data is available in BigQuery as a public data set. 

The Patient data is joined with Conditions data to append the column "prediabetic" (true/false) along with Patient's age, sex, race, marital status and zip.
```
SELECT 
      IFNULL(P.address[SAFE_OFFSET(0)].postalCode, "") AS zip,
	  sdohml.calculateAge(PARSE_DATE('%Y-%m-%d',  P.birthDate)) age, 
      P.gender sex,
      P.us_core_race.text.value.string race,
      P.maritalStatus.coding[SAFE_OFFSET(0)].code maritalStatus,
      sdohml.prediabeticcheck(coding.code) prediabetic
FROM 
      bigquery-public-data.fhir_synthea.condition  C, 
      UNNEST(C.code.coding) coding
JOIN 
      bigquery-public-data.fhir_synthea.patient P
ON 
      C.subject.patientId = P.id
WHERE 
      C.verificationstatus = 'confirmed' AND
      coding.system = 'http://snomed.info/sct'
```
## SDOH Data
Conditions in the places where people live, learn, work, and play affect a wide range of health risks and outcomes. These conditions are known as social determinants of health (SDOH). 

Along with patient data like age, race, marital status, we'll be using SDOH factors like employment, food insecurity and transportation as some of features for predicting prediabetes. Ideally, the SDOH risk factors are compiled by collecting this information from the patient and associate them with their record. [The Gravity Project](https://www.hl7.org/gravity/) is building an interoperable data standard based on HL7 FHIR for representing SDOH related data for screening, diagnosis, planning, and interventions.  The process of generating SDOH data as FHIR resources involves, Generating a FHIR Questionnaire for a specific LOINC panel like [Money and resources [PRAPARE]](https://loinc.org/93041-2/) and use the QuestionnaireResponse to build an [Observation](/LOINC-Observation.json). 

```
{
    "resourceType": "Observation",
    "code": {
        "coding": [
            {
                "code": "82589-3",
                "display": "Highest level of education",
                "system": "http://loinc.org"
            }
        ],
        "text": "Highest level of education"
    },
    "status": "final",
    "subject": {
        "reference": "Patient/6f7acde5-db81-4361-82cf-886893a3280c"
    },
    "valueCodeableConcept": {
        "coding": [
            {
                "code": "LA38-5",
                "display": "High school",
                "system": "http://loinc.org"
            }
        ],
        "text": "High school"
    }
}
```
For now, we'll just generate random responses to [Money and resources [PRAPARE]](https://loinc.org/93041-2/)  questions and bypass the creation of Observations and just append the answers/observations to out Patient data. The following view is used for generating random responses for the questions. Ideally this data should come from the Observations that were created based on the QuestionnaireResponses. 
```
CREATE OR REPLACE VIEW 
`sdohml.sdoh_questions` AS
SELECT 
STRUCT( 
    STRUCT("82589-3" as code, "Highest level of education" as display) as question, 
    4 as noOfAnswers,
    [
        STRUCT("LA30191-3" AS code,"More than high school degree" as display),
        STRUCT("LA30192-1" AS code,"High school diploma or GED" as display),
        STRUCT("LA30193-9" AS code,"Less than high school degree" as display),
        STRUCT("LA30122-8" AS code,"I choose not to answer this question" as display)
    ] as answers
) as questionSet
UNION ALL
SELECT 
STRUCT( 
    STRUCT("67875-5" as code, "Employment status current" as display) as question, 
    4 as noOfAnswers,
    [
        STRUCT("LA17956-6" AS code,"Unemployed" as display),
        STRUCT("LA30138-4" AS code,"Part-time or temporary work" as display),
        STRUCT("LA30136-8" AS code,"Full-time work" as display),
        STRUCT("LA30137-6" AS code,"Otherwise unemployed but not seeking work (ex: student, retired, disabled, unpaid primary care giver)" as display)
    ] as answers
) as questionSet
UNION ALL
SELECT 
STRUCT( 
    STRUCT("63058-2" as code, "Annual Familty Income?" as display) as question, 
    2 as noOfAnswers,
    [
        STRUCT("LA15627-5" AS code,"Less than $50,000" as display),
        STRUCT("LA15628-3" AS code,"$50,000 or more" as display)
    ] as answers
) as questionSet
UNION ALL
SELECT 
STRUCT( 
    STRUCT("93031-3" as code, "In the past year, have you or any family members you live with been unable to get any of the following when it was really needed?" as display) as question, 
    7 as noOfAnswers,
    [
        STRUCT("LA30125-1" AS code,"Food" as display),
        STRUCT("LA30126-9" AS code,"Clothing" as display),
        STRUCT("LA30124-4" AS code,"Utilities" as display),
        STRUCT("LA30127-7" AS code,"Child care" as display),
        STRUCT("LA30128-5" AS code,"Medicine or Any Health Care (Medical, Dental, Mental Health, Vision)" as display),
        STRUCT("LA30129-3" AS code,"Phone" as display),
        STRUCT("LA30122-8" AS code,"I choose not to answer this question" as display)
    ] as answers
) as questionSet
UNION ALL
SELECT 
STRUCT( 
    STRUCT("93030-5" as code, "Has lack of transportation kept you from medical appointments, meetings, work, or from getting things needed for daily living?" as display) as question, 
    4 as noOfAnswers,
    [
        STRUCT("LA30133-5" AS code,"Yes, it has kept me from medical appointments or from getting my medications" as display),
        STRUCT("LA30134-3" AS code,"Yes, it has kept me from non-medical meetings, appointments, work, or from getting things that I need" as display),
        STRUCT("LA32-8" AS code,"No" as display),
        STRUCT("LA30257-2" AS code,"Patient unable to respond" as display)
    ] as answers
) as questionSet
```
The end goal in this data prep is to have a single Patient record that contains the demographics data like age, race, sex and the SDOH factors. For example, the LOINC code for the question like [93031-3](https://loinc.org/93031-3/) will be used as a column name and the answer/observation_value_code like 'LA30125-1' will be used as the column value on the Patient record.

If you really want to build the data the right way, you can use this [Form Builder for FHIR Questionnaire](https://lhcformbuilder.nlm.nih.gov/) that generates Questionnaire for LOINC panels like [Money and resources [PRAPARE]](https://loinc.org/93041-2/). Here is a  [sample Questionnaire](/Questionnaire%20-%20Money%20and%20resources%20%5B93041-2%5D.json) generated and used as the basis for the SDOH factors in this.

For now, let's just cook up some data like this:
```
CREATE OR REPLACE VIEW 
    `sdohml.checked_patients` AS
SELECT 
    count(*) count,
    IFNULL(P.address[SAFE_OFFSET(0)].postalCode, "") AS zip,
    sdohml.calculateAge(PARSE_DATE('%Y-%m-%d',  P.birthDate)) age, 
    P.gender sex,
    P.us_core_race.text.value.string race,
    P.maritalStatus.coding[SAFE_OFFSET(0)].code maritalStatus,
    (select questionSet.answers[SAFE_OFFSET(CAST(round(rand() * (questionSet.noOfAnswers - 1)) as INT64))].code  from sdohml.sdoh_questions where  questionSet.question.code = "82589-3") as education_82589_3,
    (select questionSet.answers[SAFE_OFFSET(CAST(round(rand() * (questionSet.noOfAnswers - 1)) as INT64))].code  from sdohml.sdoh_questions where  questionSet.question.code = "67875-5") as employment_67875_5,
    (select questionSet.answers[SAFE_OFFSET(CAST(round(rand() * (questionSet.noOfAnswers - 1)) as INT64))].code  from sdohml.sdoh_questions where  questionSet.question.code = "63058-2") as annual_income_63058_2,
    (select questionSet.answers[SAFE_OFFSET(CAST(round(rand() * (questionSet.noOfAnswers - 1)) as INT64))].code  from sdohml.sdoh_questions where  questionSet.question.code = "93031-3") as prapare_survey_93031_3,
    (select questionSet.answers[SAFE_OFFSET(CAST(round(rand() * (questionSet.noOfAnswers - 1)) as INT64))].code  from sdohml.sdoh_questions where  questionSet.question.code = "93030-5") as transportation_93030_5,
    sdohml.prediabeticcheck(coding.code) prediabetic
FROM 
    bigquery-public-data.fhir_synthea.condition  C, 
    UNNEST(C.code.coding) coding
JOIN 
    bigquery-public-data.fhir_synthea.patient P
ON 
    C.subject.patientId = P.id
WHERE 
    C.verificationstatus = 'confirmed' AND
    coding.system = 'http://snomed.info/sct' 
GROUP BY
    zip,
    age,
    sex,
    maritalStatus,
    race,
    education_82589_3 ,
    employment_67875_5,
    annual_income_63058_2,
    prapare_survey_93031_3,
    transportation_93030_5,
    prediabetic
```
The above would result in data like this:
![GitHub Logo](/images/sampledata.png)
	
## Preparing Data for the model
The above generated data can be further sliced  for training, evaluating and predicting:
```
CREATE OR REPLACE VIEW
  `sdohml.input_view` AS
SELECT
    zip,
    age,
    sex,
    maritalStatus,
    race,
    education_82589_3 ,
    employment_67875_5,
    annual_income_63058_2,
    prapare_survey_93031_3,
    transportation_93030_5,
    prediabetic,
    CASE
        WHEN count > 1 and count  < 3  THEN 'training'
        WHEN count > 5 and count < 7   THEN 'evaluation'
        WHEN count > 3 and count < 5   THEN 'prediction'
    END AS dataframe
FROM
  `sdohml.checked_patients`
```
## Building the model
Now that we have prepped the training data, the next step is to create the  [logistic regression model](https://cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-create#model_option_list)  using the data. As you can see the column 'prediabetic' is the label column that would be predicted. 
```
CREATE OR REPLACE MODEL
  `sdohml.diabetes_sdoh_model`
OPTIONS
  ( model_type='LOGISTIC_REG',
    auto_class_weights=TRUE,
    data_split_method='NO_SPLIT',
    input_label_cols=['prediabetic'],
    max_iterations=15) AS
SELECT
  *
FROM
  `sdohml.input_view`
WHERE
  dataframe = 'training'
```

You can create and train a logistic regression model using the  [`CREATE MODEL`](https://cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-create)  statement with the option  `'LOGISTIC_REG'`. 

## Evaluating the model
After creating the model, evaluate the performance of the model using the  [`ML.EVALUATE`](https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-evaluate)  function. The  `ML.EVALUATE`  function evaluates the predicted values against the actual data.

The query to evaluate the model is as follows:
```
SELECT
  *
FROM
  ML.EVALUATE (MODEL `sdohml.diabetes_sdoh_model`,
    (
    SELECT
      *
    FROM
      `sdohml.input_view`
    WHERE
      dataframe = 'evaluation'
    )
  )
```
The above would result in data like this:
![GitHub Logo](/images/evaluation.png)
The accuracy of the model describes the ratio of correctly predicted values to total predictions. Please note that we generated SDOH data in a random manner which may not produce accurate predictions.  The idea of this is to show the whole pipeline of data prep (with SDOH) and using BigQuery ML. 

## Prediction
The following query predicts prediabetes for patients.
```
SELECT
  *
FROM
  ML.PREDICT (MODEL `sdohml.diabetes_sdoh_model`,
    (
    SELECT
      *
    FROM
      `sdohml.input_view`
    WHERE
      dataframe = 'prediction'
     )
  )
```
whola! Below are sample results from above prediction run. 
![GitHub Logo](/images/prediction.png)
## Summary
Here are different steps in building this: 

 1. Identify and prepare the data source: [Synthea Synthetic Patient FHIR data](https://synthetichealth.github.io/synthea/), which is also available as a public dataset that you can add to BigQuery. 
 2. Enrich Patient data with SDOH factors by leveraging Gravity/LOINC questionnaire and manufactured Observations. [Money and resources [PRAPARE]](https://loinc.org/93041-2/)
 3. Create and train a Logistic Regression model with BigQuery ML
 4. Evaluate the model using a subset of the generated data
 5. Run Prediction

As mentioned above, please note that we generated SDOH data in a random manner which may not produce accurate predictions.  The idea of this is to show the whole pipeline of data prep (with SDOH) and using BigQuery ML.

Hope this will be helpful for BigQuery ML enthusiasts who wants to leverage the technology to solve some of lingering problems in Healthcare!

