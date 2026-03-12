# ADE-extraction-with-LLM-OpenAI-o1
This project contains the python source code for using the large language model "OpenAI o1" for extracting Adverse Drug Events from a set of clinical notes.  It uses the OpenAI API to access the model hosted on Microsoft Azure. The clinical notes that were used in this model are stored in a separate physionet database project named: "Clinical Notes and LLM Output for Post-marketing Adverse Drug Event extraction with Large Language Model".  The paper that describes this project is "A Large Language Model for Extracting Post-marketing Adverse Drug Events from Clinical Notes in the Electronic Health Record" to be published in "Drug Safety" journal pending reviewer requested updates.

The accompanying manuscript is in review for publication in "Drug Safety" under the title "A Large Language Model for Extracting Post-marketing Adverse Drug Events from Clinical Notes in the Electronic Health Record"

## Background
Current post-marketing adverse event reporting requirements are mostly voluntary, and greatly underestimate the frequency of adverse events in patient experience. Regulatory authorities such as the FDA have supplemented these reports with structured medical record data such as diagnosis codes and drug orders from groups of medical centers through the FDA's Sentinel Initiative.  But structured data does not include many adverse events mentioned in clinical notes, and hence this source of adverse event signals is currently not available to regulatory authorities.

Work has previously been done on rule-based and keyword searches of clinical notes, but these are often inadequate since the methods are not aware of the context of the string matches, such as whether the adverse event was believed by the provider to be drug related. Large language models are more aware of the clinical context and have clinical knowledge that enables them to more precisely identify events that have a reasonable possibility of being drug-related.

## Software Description
The software includes the following files:

* __readme.txt__ - this file reminds the user to review the patient privacy provisions required of physionet "certified users". 

* __two_pass_o1_2026-03-11.py__. - This is the python script for implementing the two-pass large language model for extracting adverse events from clinical notes.  The text of the first and second prompts are included in the file. This implementation has been tested to work on the OpenAI service on model "o1".

* __json_schema_one.json__, __json_schema_two.json__ - These are json schema files referenced in the script, and used to define the output format of the models

* __sample_dot_env_file.txt__ - this shows the format of a required ".env" file. You will need to replace the fake values with your own values (such as OPENAI_API_KEY and file pathnames).

* __expected_standard_out.txt__ - This file is the standard output I received when executing the script "two_pass_01_2026-03-11.py" against the two fake clinical notes provided with this repo.

* __Data/In_Notes/D0001_in.txt__ and __D0002_in.txt__ - these are 2 fake clinical notes that can be safely used for testing the model, as no real patient data is contained in either of these (simplified) notes. The file name is expected to be based on the "deid_note_key" (eg, "D0001" and "D0002"). In these notes, the first line is always "__The following is a new clinical note ==========================__", but the model does not require this. Today the model expects to find a line that begins with "__patientepicid: NNNN;__", where NNNN (any length of characters before ";") is saved as the (deidentified) patient ID. The field is copied into the JSON output, but is otherwise not needed by the current LLM. The text of the actual (deidentified) note begins on the 4th line.

* __Data/Expected_out_pass_1__ - This directory contains the output I received from pass 1 for the two fake notes.  For example, for the first note, the directory contains two files:

    * __D0001_json.json__ - the JSON output for the first note, containing a list of ADEs extracted. For each ADE, there are the following fields:

        * __ade_seq__ - a sequence number for the ADE (starting with 1)
        * __Adverse Event__ - the name of the adverse event
        * __Medication__ - The name of the medication
        * __Why medication possibly caused event__ - The reason, provided by the LLM, why it believed there was a reasonable possibility that the medication caused the adverse event
        * __Verbatim phrases__ - One or more phrases from the note that were most important in the model concluding the event was an ADE

   * __D0001_out.txt__ - This is additional output generated for the note. It contains:
       * __reason_text__ - the reasoning used by the model in generating the ".json" file above
       * __response.status__ - should be "completed" if successful response
       * __in_tokens__, __out_tokens__, __reason_tokens__ - the counts of tokens created in the response. The "reason_tokens" is a subset of the "out_tokens".
       * __in token cost__, __out token cost__, __total_note_cost__ - the expected cost of the response for this note based on the "PRICE_PER_IN_TOKEN" and "PRICE_PER_OUT_TOKEN" numbers it found in your .env file. These prices can usually be found on the vendor's web server.

* __Data/Expected_out_pass_2__ - This directory contains the output I received from pass 2 for the two fake notes. For example, for the first note, the directory contains three files:

    * __D0001_in.txt__ - this is a copy of the original clinical note which should be found in directory Data/In_Notes.
    * __D0001_json.json__ - This is the JSON output for each ADE extracted. It should contain the fields from "pass_1" as well as the following additional fields:
        * __Physician Certainty__ - the physician's (note author's) level of certainty that the medication caused the adverse event
        * __LLM Certainty__ - the model's level of certainty that the medication caused the adverse event
        * __MedDRA LLT__ - the MedDRA Low Level Terminology code for the event. These are usually but not always correct.
        * __Change made__ - what change, if any, was made to the medication in response to the event.
        * __Failure of Efficacy__ - Does this event represent a failure of intended therapeutic effect? Responses are "yes", "no" or "possibly". If "yes" then the value for "unlabeled" below may not be meaningful, because most product labels don't mention failure of efficacy on their list of known adverse events.  Still, according to FDA guidelines, failure of intended therapeutic affect qualifies as an adverse event.
        * __serious__ - Is the event "serious" by US FDA criteria? Responses are "Yes", "No" or "Unk".  "Unk" would be appropriate for events that are potential serious, but there is not enough information in the note to determine if that event was serious. 
        * __reason_serious__ - Free text reason why the LLM determined the "serious" value.
        * __unlabeled__ - Is the event not found in the FDA-approved product label? Occassionally the model would make the mistake of responding "no" because the event has been reported in the scientific literature even though it has not yet been included in the product label.
        * __reason_unlabeled__ - Free text reason the LLM determined the "unlabeled" value.
        * __present_illness__ - Did the adverse event take place during the patient's present illness? This is not the same as whether or not the event was mentioned in the section of the report named "History of Present Illness". For example, some events appear in the "lab" or "physical exam" section. Some users will only want to know about events from the present illness on the assumption that this information is more detailed and reliable.
        * __event_happened_to_patient__ - (yes/no) This asks whether the event actually happened to the patient or whether it was a hypothetical problem, such as the warnings that would be included in "patient instructions" for what the medication __may__ do to a patient. This field was helpful in earlier tests with LLMs such as GPT-4o which would sometimes include hypothetical events. It wasn't a noted problem with OpenAI "o1".
        * __reason_event_happened_to_patient__ - Free text reason why the LLM determined the "event_happened_to_patient" value.
        * __med_not_specified__ - In earlier experiments with LLM GPT-4.o, the model would claim and ADE when the medication was unknown. This field was an attempt to detect that error.
        * __reason_med_not_specified__ - Free text reason why the LLM determined the "med_not_specified" value.
        * __med_used_to_treat_adverse_event__ - Was the medication used for the purpose of treating the adverse event? If "yes", it should not have been identified as an ADE. We had this problem with earlier tests with LLM GPT-4.o, but not with "o1".
        * __reason_med_used_to_treat_adverse_event__ - Free text reason why the LLM determined the "med_used_to_treat_adverse_event" value.

* __Data/Out_pass_1__ and __Data/Out_pass_2__ - These are the directories where your actual outputs from pass 1 and pass 2 will be stored when you run the script. They are empty in this repo.

## Technical Implication:
The implementation of the LLM extraction model is described in detail in the publication. The model attempts to use clinical notes to extract adverse drug event (ADE) reports suitable for post-marketed safety surveillance. As such, the model attempts to find combinations of drugs and adverse events where there is a "reasonable possibility" that the drug caused the event.

The model uses two passes to the LLM, because previous efforts to perform the objective with one task were too complicated and frequently "timed-out".

The first pass has the limited goal of just identifying the adverse events, including the drug name, event phrase, and where in the note they were found.

The second pass takes the ADEs found in the first pass and attempts to find important attributes such as "serious", "unlabeled", "failure of effectiveness", and others that may be of interest to regulatory authorities. 

## Installation and Requirements
Check with the hosting service to determine the installation requirements for the LLM API scripts, and use the "sample_dot_env_file.txt" to create your own ".env" file to decide where to place the input and output files, and to enter your API key for your account subscription.

To replicate the study with the same data as used in our study, see Physionet database "Clinical Notes and LLM Output for Post-marketing Adverse Drug Event extraction with Large Language Model" for the notes and output of the study.

IMPORTANT NOTE: This project is implemented with LLM model "OpenAI o1", which is the same model used in the manuscript. In the event that the model is no longer supported by OpenAI, we have adapted the code to be run under model "gpt-5.2" and published the adapted code at Physionet.org, in software project "Source Code for Post-marketing Adverse Drug Event extraction with Large Language Model". If you chose to run the software under "gpt-5.2" with the same set of deidentified notes as used in the study, your results may be different and hopefully better.  

## Usage Notes
Use the following steps to extract ADEs from a set of notes:

* 

```
python3 two_pass_o1_2026-03-09.py.
```

## Acknowledgements
The authors thank Atul Butte and Travis Zack for their valuable guidance and support in this project.  We thank the UCSF AI Tiger Team, Academic Research Services, Research Information Technology, and the Chancellor’s Task Force for Generative AI for their software development, analytical and technical support related to the use of Versa API Gateway (the UCSF secure implementation of large language models and generative AI via API gateway).
