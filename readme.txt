readme.txt for sample code
by Dana Ludwig
12/15/2025

Be sure to read and comply with the terms of the use of physionet in protecting patient data,
and the credentialed data use agreement, as currently shown at this URL:

https://physionet.org/news/post/llm-responsible-use

This sample code is close to the sample code used in the journal paper
with the following exceptions:

* The paper uses Azure's version of OpenAI o1, but this uses OpenAI's
  hosting server. There are slight differences in connecting to the model
* The journal paper uses the "completions" API, and this sample code
  uses the "responses" API. This is because OpenAI states that the
  responses API is newer and can give a bit better performance
* The journal paper gets json output through parameter
    response_format={ "type":"json_object"}
  This sample code uses the parameters:
    text={
      "format": {
         "type": "json_schema",
         "name": "notes",
         "strict":True,
         "schema": json_schema_one (or json_schema_two)
       }
     }
  Where "json_schema_one" and "json_schema_two" are provide to you
  in the two files "json_schema_one.json" and "json_schema_two.json"
  This was done because GPT-5.2 "responses" API does not accept the
  previous parameters for json output.
* This sample code supports saving the output of the reasoning summaries,
  if any, in the "*_out.txt" output text files. The reassoning is
  interesting and may be helpful in future improvements to model.
* This sample code generates a slightly different output JSON format 
  than the code published in the paper, and shown in the dataset json files.
  - The paper and files have a format:
    [
      {
        "patientepicid": "PPPPP",
        "died_note_key": "AAAAA",
        "Adverse Event 1": {
           "Adverse Event": "aaaaaaa",
           "Medication": "mmmmmmmm",
           ... etc.
        },
        "Adverse Event 2": {
           ... etc.
        }
      }
    ]
  - This code generates files with a format:
    {
        "patientepicid": "PPPPP",
        "died_note_key": "AAAAA",
        "ADEs": [
           {
           "ade_seq": 1,
           "Adverse Event": "aaaaaaa",
           "Medication": "mmmmmmmm",
           ... etc.
           },
           {
           "ade_seq": 2,
           ... etc.
           }
        ]
    }
  The reasons for this code change were:
  - The original code could not be described in a json schema due to the
    variable and unlimited number of field names: 
       "Adverse Event 1", "Adverse Event 2", etc.
  - The OpenAI structured data rules required that the schema be a "dictionary"
    and not a "list", at the top level.
   
  
