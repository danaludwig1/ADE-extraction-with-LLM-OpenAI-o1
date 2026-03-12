##############################
#
# REPLACE THESE VALUES WITH VALUES FROM YOUR OWN ENVIRONMENT
from dotenv import load_dotenv, find_dotenv
import os
# puts OPENAI_API_KEY in your ".env"
# so the OpenAI() call can find it.
_ = load_dotenv(".env")    
# You will need other variables in the .env file; 
# see sample .env file for parameters and change values to your values
model_id = os.getenv("MODEL_ID")
price_per_in_token = float(os.getenv("PRICE_PER_IN_TOKEN"))
price_per_out_token = float(os.getenv("PRICE_PER_OUT_TOKEN"))
out_dir_prefix = os.getenv("OUT_DIR_PREFIX")
out_dir_1_pass = os.getenv("OUT_DIR_1_PASS")
out_dir_2_pass = os.getenv("OUT_DIR_2_PASS")
in_dir_prefix  = os.getenv("IN_DIR_PREFIX")
in_note_dir = os.getenv("IN_NOTE_DIR")
dir_delimiter = os.getenv("DIR_DELIMITER")
sleep_secs = int(os.getenv("SLEEP_SECS"))
#############################################################

# Two pass final version on OpenAI
import json
import matplotlib.pyplot as plt
import numpy as np
import openai
import os
import pandas as pd
import pickle
import pytz
import sys
import time
import urllib
import subprocess
import re
import tiktoken
import re
import copy
import traceback
import shutil

from datetime import datetime
from IPython.display import Markdown, Image
from openai import OpenAI
# from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm import tqdm
# from sqlalchemy import create_engine, text
from typing import Union, List

##########################

# current_key = os.environ.get('OPENAI_API_KEY')
# print("current key:", current_key)

# encoding = tiktoken.encoding_for_model(model_id.lower())

######################################

# This is the first prompt
system_text_pass1_v4 = """
1) Read the note or notes carefully. Identify the patientepicid and deid_note_key for that note from the note header.
2) Identify all pairs of medication and one or more adverse events that may be caused by the medication.
Name as many adverse events, both serious and non-serious, as are found in the block of notes.

Exclude any mention about hypothetical events, that is, possible adverse events that didn't 
actually happen to the patient.
for instance in instructions to patients, if the provider warns the patient, 
"You may expect to: have a headache. Have nausea.", 
then headache and nausea are NOT adverse events, and should be excluded, 
even if they are mentioned on separate sentences.

Medications include prescription drug, over the counter drugs, and biologics (eg, medicinal proteins, immune globulins or monoclonal antibodies)
Medications do not include medical devices or procedures, dietary supplements, foods or cosmetics 
or deficiency states in the patient (eg, Vitamin B12 deficiency).
Adverse events include any undesirable experience associated with the use of a medication, as defined above. 
The Adverse Events can include: 
    * Unintended signs, symptoms, or diseases
    * Abnormal laboratory findings
    * Drug overdoses, whether accidental or intentional
    * Drug abuse
    * Failure of expected pharmacological action

    Exclude any adverse events that don't have at least one medication and one adverse event.<br>
    Exclude any adverse events that do not have a reasonable possibility that the medication caused the adverse event.<br>
    BUT DO include the event and medication if you do think there is a reasonable possibility of causality even if:
      * the physician did not suggest that the medication may have caused the event, or
      * the adverse event has never been mentioned in the scientific literature or FDA-approved prescribing information.
    
    This is because the FDA is very interested in discovering new, previous unknown adverse events to medications.
    
    The adverse event should be specified as precisely and narrowly as possible. For example if the patient has an adverse event of "urticary",
    then specify "Adverse event" as "urticary" and not "Mild Allergic Transfusion Reaction".<br>
    If the reporter does not explicitly call out the possibility of the medication causing the event, and if the adverse event
    was the reason (ie, indication) for the physician to prescribe (or raised the dose) of the drug,<br>
    then exclude that adverse event from the output.<br>
    If there is no medication as the possible cause of the adverse event, 
    then exclude the adverse event from the output.<br>
    If no medicine trade name or generic name is specified but only a class of medications is specified 
    (e.g., muscle relaxant, cancer chemotherapy) then exclude the adverse event.<br>
    The "medication" field must not be a disease or procedure as the possible cause of the event.<br>
    If the only cause of the event is a disease or procedure, then exclude the adverse event from the output.<br>
    If the medication is used prophylactically but fails to prevent the event, include this adverse event and flag it as "Failure of efficacy" = "yes"<br>
    If the medication as used always fails to even partially resolve the event, include this adverse event and flag it as "Failure of efficacy" = "yes".<br>
    However, if the medication fails on the initial dose, but at least partially resolves the event after raising the dose, eliminate the adverse event.<br>

3) Why do you think the medication may have caused the adverse event?  If you don't have a reason, then
eliminate the adverse event. The answer should be free text and as long as you need to give your reasons.

4) "Verbatim phrases": record a list of verbatim phrases quoted from the note that show why you think 
the medication possibly caused the adverse event.
Provide a complete list of all phrases from the note text that 
supported your conclusion. If possible, have a verbatim phrase that shows that the reporter 
mentioned the drug as a possible cause of the event in the same verbatim phrase.

5) Please present the findings in a JSON format as shown in the example below.
The value of "ade_seq" should be a sequence number starting with 1, 
then 2,3,etc.

{
\n"patientepicid": "the patientepicid found at the top of the first note",
\n"deid_note_key": "the deid_note_key from the header of the note containing this adverse event",
\n"ADEs": [ 
\n    {
\n    "ade_seq":1,
\n    "Adverse event": "Description of symptom, lab abnormality, or new diagnosis",
\n    "Medication": "medication associated with the adverse event. Never use a deficiency state, disease or procedure",
\n    "Why medication possibly caused event": "free text reasons for including this as an ADR",
\n    "Verbatim phrases": [{"phrase1","phrase2","phrase3"}],
\n    },
\n    {
\n    "ade_seq":2,
\n    ... etc, for as many adverse events as you find in this note.
\n    }
\n]
\n}

When no adverse events are found for the note, just output the three JSON variables "patientepicid" and "deid_note_key" and "ADEs" and have "[]" as the value 
for "ADEs".

Please attempt to extract 20 or more adverse events with each execution
"""

#########################################

# This is the first pass
def extract_one_note_pass1_9(out_dir, in_note_path, tot_in_tokens, \
    tot_out_tokens, out_json_path, out_txt_path, deid_note_key, note_text, \
    system_text, effort, timeout, max_retries):
    # The ADE extraction is broken up into two passes:
    #   - query_and_extract_pass1_9() - this script; the first pass script
    #   - query_and_extract_pass2_10() - The second pass script
    # This function analyzes the current single note and appends it to the json_str
    # It also appends the note and the diagnostic text to the two output files

    # patientepicid = note_text.split("patientepicid: ")[1]
    # patientepicid = patientepicid.split(";")[0]
    # print("patientepicid=", patientepicid)
    # with open(in_note_path, "a") as f: f.write(note_text+'\n')
    with open("json_schema_one.json", "r") as file:
        json_schema_one = json.load(file)

    system_prompt1 = "You are an physician expert researching adverse events associated with any drugs or biologics."
    user_prompt1 = system_text + "\n\n```" + note_text + "\n```"
    client = OpenAI()
    exception_found = False
    error_out = ""
    refusal = ""
    collected_chunks = []
    in_tokens=0; out_tokens=0; reason_tokens=0
    print("responses started at", datetime.now())
    try:
        response = client.responses.create(
            model=model_id,
            instructions=system_prompt1,
            input=user_prompt1,
            reasoning={"effort":"high", "summary":"auto"},
            store=False,                   # To protect patient privacy
            text={
                "format": {
                "type": "json_schema",
                "name": "notes",
                "strict": True,
                "schema": json_schema_one
              }
            }, 
            # seed = 12345,                   # to make more reproduceable
        )
    except Exception as err:
        exception_found = True
        error_out = f"extract_one_note_pass1_9() on responses.create() caught error {err=}, Type {type(err)=}"
        print(error_out)
        print("time failed", datetime.now(), "; tokens before this note:")
        # print("Input value of user_prompt1 was: ===========================")
        # print(user_prompt1)
        # print("============================================================")
        print("tot_in_tokens = ", tot_in_tokens, "; tot_out_tokens = ", tot_out_tokens, "price_per_in_token", price_per_in_token)
        print("cost for model ", model_id, "; prompt= $"+ str(tot_in_tokens*price_per_in_token) \
                    + "; completion= $" + str(tot_out_tokens*price_per_out_token) \
                    + "; TOTAL = $" + str(tot_in_tokens*price_per_in_token \
            + tot_out_tokens*price_per_out_token) + "\n")        
    if not exception_found:
        print("responses returned at", datetime.now())
        parsed_json = json.loads(response.output_text)
        json_str = json.dumps(parsed_json, indent=4)
        in_tokens = response.usage.input_tokens
        out_tokens = response.usage.output_tokens
        reason_tokens = \
            response.usage.output_tokens_details.reasoning_tokens
        reason_text = ""

        # This parsing of reasoning summary may not work until you have validated
        # the identify of your "organization" with OpenAI. For me, this simply
        # required me giving a digital photo of my driver's license and letting
        # their app make a photo of my face, and then their software verifies
        # that the two photos are the same person.  
        # go to URL: https://platform.openai.com/settings/organization/general
        # and in the center of page, look for Organization Settings -> 
        # Verifications -> click "Verify Organization"
        # You may need to wait 24 hours before the "reasoning" feature is activated
        # If you can't do that, then
        # Just delete the "reasoning" parsing lines below.

        for output_item in response.output:
            if output_item.type == 'reasoning':
                for content_item in output_item.summary:
                    reason_text += '\n'+content_item.text

        # print("result is json_str:")
        # print(json_str)
        with open(out_json_path, "a") as f:
            f.write(json_str)
        with open(out_txt_path, "a") as f:
            f.write("deid_note_key: " + deid_note_key + "\n") 
            f.write("\nreason_text:\n" + reason_text+"\n\n")
            f.write( \
                    "response.status: "+response.status+ \
                    "in_tokens: " + str(in_tokens) + "; out_tokens: " + str(out_tokens)+ \
                    "; reason_tokens: " + str(reason_tokens)+"\n" )
    else:    # an exception was found
        with open(out_txt_path, "a") as f:
            f.write("deid_note_key: " + deid_note_key + "\n") 
            f.write(error_out + "\n")
            f.write("time failed "+str(datetime.now()))
    return (in_tokens, out_tokens, reason_tokens)

def query_and_extract_pass1_9(patientepicid_list, out_dir, system_text, \
    in_note_path, note_list = "", effort="medium", timeout=5600, \
    just_note_list=False, max_retries=6):
    # This started as a complete copy of query_and_extract8 to make the naming of the first pass consistent with the 2nd pass
    #   - query_and_extract_pass1_9() - this script; the first pass script
    #   - query_and_extract_pass2_10() - The second pass script
    # We want one file per patient. Each file will have one or more notes.
    print("Starting Pass 1 ==============================")
    isExist = os.path.exists(out_dir)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(out_dir)    
    tot_in_tokens = 0
    tot_out_tokens = 0
    tot_reason_tokens = 0
    err_note_key_list = []
    
    json_str = ''                 # Accumulated json output from note.

    entries = os.listdir(in_note_path)
    entries.sort()
    for infile in entries:
        if infile[-7:] != "_in.txt":
            continue
        my_note_path = in_note_path +dir_delimiter+infile
        with open(my_note_path, "r", encoding="Latin-1") as f:
            new_note_text = f.read()
        new_deid_note_key = infile.split('_in')[0]
        out_json_path = out_dir+dir_delimiter+new_deid_note_key+"_json.json"
        out_txt_path = out_dir+dir_delimiter+new_deid_note_key+"_out.txt"
        with open(out_txt_path, "w") as f: f.write("")
        with open(out_json_path, "w") as f: f.write("")
        print("about to extract note", new_deid_note_key, \
            "; time is", datetime.now())
        in_tokens, out_tokens, reason_tokens = \
            extract_one_note_pass1_9(out_dir, my_note_path, tot_in_tokens, \
            tot_out_tokens, out_json_path, out_txt_path, \
            new_deid_note_key, new_note_text, system_text, effort, \
            timeout, max_retries)
        if in_tokens == 0:          # exception
            err_note_key_list.append(new_deid_note_key)
        if infile != entries[-1]:
            print("completed note", new_deid_note_key, "at", datetime.now(), \
                "; waiting", sleep_secs, "seconds")
            time.sleep(sleep_secs) # wait <sleep_secs> seconds to avoid timeout
        else: 
            print("completed last note", new_deid_note_key, "at", \
              datetime.now())
        cost_in = in_tokens*price_per_in_token
        cost_out = out_tokens*price_per_out_token
        with open(out_txt_path, "a") as f: 
            f.write("in token cost=$"+str(cost_in)+"out token cost=$"+ \
                str(cost_out)+"; total note cost=$"+str(cost_in+cost_out)+"\n")
        tot_in_tokens += in_tokens
        tot_out_tokens += out_tokens
        tot_reason_tokens += reason_tokens
        with open(out_json_path, "a") as f: f.write("\n")

    print("tot_in_tokens = ", tot_in_tokens, "; tot_out_tokens = ", \
         tot_out_tokens, "; tot_reason_tokens = ", tot_reason_tokens)
    print("price_per_out_token=", price_per_out_token, "price_per_in_token=", \
         price_per_in_token)
    tot_out = tot_out_tokens*price_per_out_token
    tot_in = tot_in_tokens*price_per_in_token
    tot_both = tot_out + tot_in
    print("tot_out, tot_in, tot_both are", tot_out, tot_in, tot_both)
    print("cost for model ", model_id, "; in=", tot_in, "; out=", tot_out, "; TOTAL=", tot_both)
    if len(err_note_key_list) > 0:
        print("Failed notes (if any):")
        print(err_note_key_list)
        for i in range(0, len(err_note_key_list), 5):
            err_note_string = "'" + "', '".join(err_note_key_list[i:i+5]) \
                + "' "+dir_delimiter
            print(err_note_key_list)

########################################################

# Prompt for pass 2
system_text_pass2_v5 = """
Below is a list of generated Adverse Drug Events, in JSON format, followed by a single clinical note from which all the 
events were generated. For each adverse event, I want you to assume that it is correct, 
and then study the clinical note to add additional fields (described below) to each event.

In cases where the JSON file contains no adverse events, but just the fields "patientepicid" and "deid_note_key", 
then pass these 2 fields to the output and quit.

1) For JSON files with 1 more more adverse events, please generate the additional fields for each input adverse event at a time.
It is critically imporptant that you generate these fields for __ALL__ adverse event in the input, and not just some of them.

2) The input adverse events will each have at least the following fields:
    
    2.1) Adverse event - this is what went wrong with the patient
    
    2.2) Medication - this is the medication that was judged to be a possible cause of the adverse event
    
    2.3) Why medication possibly caused event - this is the reason that the previous expert or LLM believed that
    the medication was a possible cause of the adverse event.  Assume this is a good reason.

    2.4) Verbatim phrases - this is a (python) list of verbatim phrases quoted from the note that show why the expert
    or LLM thought the medication possibly caused the adverse event.
    
3) Please study the note and add the following fields to each adverse event in the input:
    
    3.1) Physician Certainty - Assess the reporting physician’s level of certainty regarding whether this medication 
    caused the adverse event or toxicity. Choose one of the following options:
    
        * Discaused as the cause of event
        * Discussed as a potential cause
        * not mentioned as cause by physician
    
    3.2) LLM Certainty - provide your (Large language model) level of certainty as to whether the medication 
    caused the event. Options are:
    
        * Certain of causality
        * potential cause
        * unlikely cause
    
    3.4) MedDRA LLT - Using the event phrase and the verbatim phrases, determine the best 
    MedDRA Lowest Level Term (LLT) code for the event.

    3.5) Change made - Indicate if any change was made to the medication in response to the 
    adverse event or toxicity. Options are:
    
        * No change made
        * Dose modified
        * Drug held
        * Drug permanently discontinued
        * Unclear

    3.6) Failure of Efficacy - Determine whether this potential adverse event represents a failure of expected 
    efficacy of the drug. Options are:
    
        * Yes  (this means that the adverse event started before the drug was administered, and the drug is typically used
        to treat the event, but the event was not relieved or reduced by the drug. This does not include cases
        where the event was increased in severity after the drug was administered because in that case the drug
        may be causing the event or making it worse.)
        * Possibly (same as 'Yes', but you are not sure whether the drug failed its expected efficacy)
        * No

    3.7) Serious - Determine whether this adverse event was serious, by FDA standards. Options are Yes, No or Unk.
    At URL "https://www.fda.gov/safety/reporting-serious-problems-fda/what-serious-adverse-event" 
    The FDA defines a serious adverse event as follows:

        The event is serious when the patient outcome is:
    
        * Death - You suspect that the death was an outcome of the adverse event, and include the date if known. 
        * Life-threatening - You suspected that the patient was at substantial risk of dying at the time of the adverse event, 
        or use or continued use of the device or other medical product might have resulted in the death of the patient.
        * Hospitalization (initial or prolonged) - Admission to the hospital or prolongation of hospitalization 
        was a result of the adverse event.
        * Emergency room visits that do not result in admission to the hospital should be evaluated for one of the other serious outcomes 
        (e.g., life-threatening; required intervention to prevent permanent impairment or damage; other serious medically important event).
        * Disability or Permanent Damage - The adverse event resulted in a substantial disruption of a person's ability to conduct normal life functions, 
        i.e., the adverse event resulted in a significant, persistent or permanent change, impairment, damage or disruption in the 
        patient's body function/structure, physical activities and/or quality of life.
        * Congenital Anomaly/Birth Defect - You suspect that exposure to a medical product prior to conception or during pregnancy may have resulted 
        in an adverse outcome in the child.
        * Other Serious (important medical events) - The event does not fit the other outcomes, but the event may jeopardize the patient 
        and may require medical or surgical intervention (treatment) to prevent one of the other outcomes. 
        Examples include allergic brochospasm (a serious problem with breathing) requiring treatment in an emergency room, 
        serious blood dyscrasias (blood disorders) or seizures/convulsions that do not result in hospitalization. 
        The development of drug dependence or drug abuse would also be examples of important medical events.

    3.8) reason_serious - Why do you think the medication may have caused the adverse event?  The answer should be free text 
    and as long as you need to give your reasons.
    
    3.9) unlabeled - Determine whether this adverse event was not already contained in the FDA-approved prescribing information. 
    Options are Yes or No.
    
        * Use "no" if an FDA product label mentions the adverse event or an event with approximately 
        equivalent meaning in sections "Warnings and Precautions", "Adverse Reactions" or "Drug Interactions". 
        * Use "yes" if the FDA doesn't mention this event or an equivalent event in these three sections
        * use "yes" IF the adverse event has been reported in the medical literature, but is not mentioned in the FDA-approved prescribing information,
        then use the value Yes, meaning that the event is unlabeled
    
    3.10) reason_unlabeled - Why do you think the medication is or is not unlabeled? The answer should be free text 
    and as long as you need to give your reasons. If unlabeled is 'No', give a URL reference to the FDA-approved prescribing information
    for the medication that mentions the adverse event.
    
    3.11) present_illness - Was this event part of the current complaint, history of present illness, review of systems,
    phyical exam, lab values, or the current procedure?  If so, enter "yes".
    If it was only part of the past medical history, family history or allergies section, enter "no".
    If unknown, enter "unk".
    
    3.12) event_happened_to_patient - Did the adverse event actually happen to the patient as of the writing of this note?  Value is 'no'
    if the event did not happen to the patient or did not happen yet. For example:
    
        * the adverse event happened to a family member in family history
        * In patient instruction, if a medication is prescribed, it is typical that the instructions will name all the side
        effects that the patient may experience while taking the medication. The key word is "may". If you see that word in the 
        context of patient instructions, you must say 'no', the event did not (yet) happen to the patient.
    
    If the event did actually happen to the patient, answer 'yes'
    
    3.13) reason_event_happened_to_patient - why do you believe that the event did or did not happen to the patient?
    
    3.14) med_not_specified - was there no medication specified? If a specific medication was not specified in the note, 
    enter "yes", otherwise enter "no".
    
    3.15) reason_med_not_specified - why did you conclude the med_not_specified was yes or no?
    
    3.16) med_used_to_treat_adverse_event - if the med was used to treat the adverse event 
    and the addition or raising the dose of the med did not make the event worse, enter "yes". Otherwise enter "no"
    
    3.17) reason_med_used_to_treat_adverse_event - why did you conclude med_used_to_treat_adverse_event was yes or no?

4) Please present the findings in a JSON format as shown below:
JSON Output Format Example:

{
\n"patientepicid": "the patientepicid found at the top of the first note",
\n"deid_note_key": "the deid_note_key from the header of the note containing this adverse event",
\n"ADEs": [
\n    {
\n    "Adverse event": "Description of symptom, lab abnormality, or new diagnosis",
\n    "Medication": "medication associated with the adverse event. Never use a deficiency state, disease or procedure",
\n    "Why medication possibly caused event": "free text reasons for including this as an ADR",
\n    "Verbatim phrases": [{"phrase1","phrase2","phrase3"}],
\n    "Physician Certainty": "Physician Certainty level from options above",
\n    "LLM Certainty": "Your (large language model) Certainty level from options above",
\n    "MedDRA LLT": "You determination of the best LLT for the adverse event"
\n    "Change made": "Change, if any, from options above",
\n    "Failure of Efficacy": "Your judgement that the event represents failure of expected efficacy from options above",
\n    "serious":"yes, no or unk",
\n    "reason_serious":"your reason for the serious value",
\n    "unlabeled":"yes or no",
\n    "reason_unlabeled": "your reason for the unlabeled value",
\n    "present_illness":"yes, no or unk",
\n    "event_happened_to_patient":"yes or no",
\n    "reason_event_happened_to_patient":"your reason for the hypothetical value",
\n    "med_not_specified":"yes or no",
\n    "reason_med_not_specified":"your reason for the med_not_specified value",
\n    "med_used_to_treat_adverse_event":"yes or no",
\n    "reason_med_used_to_treat_adverse_event":"your reason for the med_used_to_treat_adverse_event value"
\n    },
\n    {
\n    "ade_seq":2,
\n    ... etc, for as many adverse events as you find in this note.
\n    }
\n]
\n}

When no adverse events are found, just output the two JSON variables "patientepicid" and "deid_note_key" and "ADEs" and have "[]" as the value for "ADEs".
"""
# print("system_text_pass2_v5:")
# print(display(Markdown(system_text_pass2_v5)))
                   
#####################################

def extract_one_note_pass2_10(out_dir, in_ae_cnt, in_json_path, in_note_path, tot_in_tokens, tot_out_tokens, out_json_path, out_txt_path, \
        out_note_path, patientepicid, deid_note_key,json_text, note_text, system_text, effort, timeout, max_retries, debug):
    # This function analyzes the current single note and appends it to the json_str
    # It also appends the note and the diagnostic text to the two output files
    with open(out_note_path, "a") as f: 
        f.write(note_text)
    system_prompt1 = "You are an physician expert researching adverse events associated with any drugs or biologics."
    user_prompt1 = system_text + "\n\nAdverse Events in JSON:\n\n```\n"+json_text+ \
        "\n```\n\nClinical Note Text:\n\n```\n" + note_text + "\n```"
    # user_prompt1_token_cnt = len(encoding.encode(user_prompt1))
    # system_prompt1_token_cnt =  len(encoding.encode(system_prompt1))
    # system_text_token_cnt = len(encoding.encode(system_text))
    # json_text_token_prompt = len(encoding.encode(json_text))
    # note_text_token_prompt = len(encoding.encode(note_text))
    # print("user_prompt1_token_cnt is:", user_prompt1_token_cnt)
    # print("  ; system_prompt1=", system_prompt1_token_cnt, "; system_text=", system_text_token_cnt, "; json_text=", json_text_token_prompt, \
    #        "; note_text =", note_text_token_prompt)
    with open("json_schema_two.json", "r") as file:
        json_schema_two = json.load(file)

    client = OpenAI()

    exception_found = False
    error_out = ""
    try:
        response = client.responses.create(
            model=model_id,
            instructions=system_prompt1,
            input=user_prompt1,
            reasoning={"effort":"high", "summary":"auto"},
            store=False,                   # To protect patient privacy
            text={
                "format": {
                "type": "json_schema",
                "name": "notes",
                "strict": True,
                "schema": json_schema_two
              }
            },
        )
    except Exception as err:
        exception_found = True
        error_out = f"extract_one_note_pass2_10() on responses.create() caught error {err=}, Type {type(err)=}"
        print(error_out)
        print("time failed", datetime.now(), "; tokens before this note:")
        print("tot_in_tokens = ", tot_in_tokens, "; tot_out_tokens = ", tot_out_tokens, \
            "price_per_in_token", price_per_in_token)
        print("cost for model ", model_id, "; prompt= $"+ \
            str(tot_in_tokens*price_per_in_token) \
            + "; completion= $" + str(tot_out_tokens*price_per_out_token) \
            + "; TOTAL = $" + str(tot_in_tokens*price_per_in_token \
            + tot_out_tokens*price_per_out_token) + "\n")        
    if not exception_found:
        print("responses returned at", datetime.now())
        out_parsed_json = json.loads(response.output_text)
        out_json_str = json.dumps(out_parsed_json, indent=4)
        in_tokens = response.usage.input_tokens
        out_tokens = response.usage.output_tokens
        reason_tokens = \
            response.usage.output_tokens_details.reasoning_tokens
        reason_text = ""

        # This parsing of reasoning summary may not work until you have validated
        # the identify of your "organization" with OpenAI. For me, this simply
        # required me giving a digital photo of my driver's license and letting
        # their app make a photo of my face, and then their software verifies
        # that the two photos are the same person.
        # go to URL: https://platform.openai.com/settings/organization/general
        # and in the center of page, look for Organization Settings ->
        # Verifications -> click "Verify Organization"
        # You may need to wait 24 hours before the "reasoning" feature is activated
        # If you can't do that, then
        # Just delete the "reasoning" parsing lines below.

        for output_item in response.output:
            if output_item.type == 'reasoning':
                for content_item in output_item.summary:
                    reason_text += '\n'+content_item.text

        with open(out_json_path, "a") as f:
            f.write(out_json_str)
        with open(out_txt_path, "a") as f:
            f.write("deid_note_key: " + deid_note_key + "\n")
            f.write("\nreason_text:\n" + reason_text+"\n\n")
            f.write( \
                "response.status: "+response.status+ \
                "in_tokens: " + str(in_tokens) + "; out_tokens: " + str(out_tokens)+ \
                "; reason_tokens: " + str(reason_tokens)+"\n" )
            f.write("total tokens (in+out) = "+ str(in_tokens + out_tokens) + "\n")
            f.write("cost for model '"+model_id+"': prompt= $"+ str(in_tokens*price_per_in_token) \
                    + "; completion= $" + str(out_tokens*price_per_out_token) \
                    + "; TOTAL = $" + str(in_tokens*price_per_in_token \
                    + out_tokens*price_per_out_token) + "\n")
        ades = out_parsed_json["ADEs"]
        out_ae_cnt = len(ades)
        print("in_ae_cnt=", in_ae_cnt, "; out_ae_cnt=", out_ae_cnt)
        if out_ae_cnt < in_ae_cnt:
            print("ERROR IN LLM FOR NOTE", deid_note_key)
            print("pass 2 did not process all ADEs found in pass 1")
            print("ae_cnt from pass 1=", in_ae_cnt, "; from pass 2=", out_ae_cnt)
            print("This note will need to have pass 2 re-run.")
        print("note completed at", datetime.now())
    else:    # an exception was found
        in_tokens = 0
        out_tokens = 0
        reason_tokens = 0
        with open(out_txt_path, "a") as f:
            f.write("patientepicid: " + patientepicid + "; deid_note_key: " + deid_note_key + "\n") 
            f.write(error_out + "\n")
            f.write("time failed "+str(datetime.now()))
    return (in_tokens, out_tokens, reason_tokens)
        

def query_and_extract_pass2_10(in_dir, out_dir, note_dir, system_text, effort="medium", \
  timeout=5600, max_retries=6, only_note_file="", debug=False):
    print("Starting Pass 2 ==============================")
    # We want one file per patient. Each file will have one or more notes.
    isExist = os.path.exists(out_dir)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(out_dir)    
    tot_note_cnt = 0
    tot_in_tokens = 0
    tot_out_tokens = 0
    tot_reason_tokens = 0
    err_note_key_list = []
    ae_cnt = 0
    cols = 'ntokens'
    ordered_note_list = []
    ordered_note_string = ""
    new_in_tokens_tot_num = 0
    context_num = 0
    new_deid_note_key_list = []
    json_str = ''                 # Accumulated json output from each note.
    file_list = []
    in_note_list = []
    out_note_list = []
    json_struct = []
    valid_dict = {"ade_valid":"", "ade_valid_reason":"",  "serious_valid":"", "unlabeled_valid":"","present_illness_valid":""}
    file_list = []
    # in_dir = in_dir_prefix + dir_delimiter + dir_suffix
    for file in os.listdir(in_dir):
        if file.endswith(".json") and file[0] != ".":
            file_list.append(file)
    # First process the json input file
    for file in file_list:
        in_tokens = 0
        out_tokens = 0
        reason_tokens = 0
        # need to use "rb" because o1 generates smart apostrophe's 
        # that can't read() as string.
        with open(in_dir + dir_delimiter + file, "rb") as infile:
            # print("about to load input json file", in_dir + dir_delimiter + file)
            in_ae_cnt = 0
            tot_note_cnt += 1
            if only_note_file != "" and file != only_note_file:
                continue
            in_json_path = in_dir + dir_delimiter + file
            in_note_path = note_dir + dir_delimiter \
                + file.replace("_json.json", "_in.txt")
            out_note_path = out_dir + dir_delimiter + \
                file.replace("_json.json", "_in.txt")
            out_txt_path = out_dir + dir_delimiter + \
                file.replace("_json.json", "_out.txt")
            out_json_path = out_dir + dir_delimiter + file
            json_bytes = infile.read()
            # now convert any "’" character to "'"
            json_bytes = json_bytes.replace(b"\x92", b"'")  # get rid of o1 generated smart apostrophes
            json_bytes = json_bytes.replace(b"\x85", b".")  # get rid of o1 generated "..." char
            json_bytes = json_bytes.replace(b"\x91", b"'")  # get rid of o1 generated left smart quote
            json_bytes = json_bytes.replace(b"\x93", b"'")  # get rid of o1 generated right smart double quote
            json_bytes = json_bytes.replace(b"\x94", b".")  # get rid of o1 generated "cancel character" that chatGPT uses to indicate a reference
            json_bytes = json_bytes.replace(b"\x96", b"-")  # get rid of o1 generated "en dash" 
            # json_bytes = json_bytes.replace(b"\u2011", b"-")  # get rid of o1 generated Unicode \u2011 (Non-Breaking Hyphen):
            json_bytes = json_bytes.replace(b"\x97", b" ")  # get rid of o1 generated END OF GUARDED AREA (need it because this has encoded password)
            json_text = str(json_bytes)
            exception_found = False
            try:
                note_dict = json.loads(json_bytes)
            except Exception as err4:
                exception_found = True
                error_out = f"extract_one_note_pass2_10() on completions.create() caught error {err4=}, Type {type(err4)=}"
                print(error_out, "; note_key =", file)
                print("json_bytes =========================================")
                print(json_bytes)
                print("====================================================")
                print("time failed", datetime.now(), "; tokens before this note:")
                print("tot_in_tokens = ", tot_in_tokens, "; tot_out_tokens = ", tot_out_tokens)
                print("cost for model ", model_id, "; prompt= $"+ str(tot_in_tokens*price_per_in_token) \
                            + "; completion= $" + str(tot_out_tokens*price_per_out_token) \
                            + "; TOTAL = $" + str(tot_in_tokens*price_per_in_token \
                            + tot_out_tokens*price_per_out_token) + "\n") 
                print("=========================================")
                print("")
                print("stack dump to follow")
                raise
            if not exception_found:
                patientepicid = note_dict["patientepicid"]
                deid_note_key = note_dict["deid_note_key"]
                ade_list = note_dict["ADEs"]
                # find out how many input ADE's there are for this note
                in_ae_cnt = len(ade_list)
                if in_ae_cnt == 0:             # Just copy existing files to output
                    shutil.copy(in_json_path, out_dir)
                    shutil.copy(in_note_path, out_dir)
                    with open(out_txt_path, "w") as f:
                        f.write("deid_note_key "+deid_note_key+ \
                            " had zero ADEs so skipped LLM pass 2\n")
                else:
                    with open(in_note_path, "r", encoding="Latin-1") as f: 
                        note_text = f.read()
                    with open(out_txt_path, "w") as f: f.write("")
                    with open(out_note_path, "w") as f: f.write("")
                    with open(out_json_path, "w") as f: f.write("")
                    print("about to extract note", deid_note_key, "; time is", datetime.now())
                    in_tokens, out_tokens, reason_tokens = \
                        extract_one_note_pass2_10(out_dir, in_ae_cnt, in_json_path, \
                           in_note_path, tot_in_tokens, tot_out_tokens, out_json_path, \
                           out_txt_path, out_note_path, patientepicid, deid_note_key, \
                           json_text, note_text, system_text, effort, timeout, max_retries, \
                           debug)
                    if in_tokens == 0:          # exception
                        err_note_key_list.append(deid_note_key)
                    cost_in = in_tokens*price_per_in_token
                    cost_out = out_tokens*price_per_out_token
                    with open(out_txt_path, "a") as f: 
                        f.write("in token cost=$"+str(cost_in)+"out token cost=$"+ \
                            str(cost_out)+"; total note cost=$"+str(cost_in+cost_out)+"\n")
                try:                  # checking for bad unicode chars
                    with open(out_json_path, "a") as f: 
                        f.write("\n")
                except Exception as err3:
                    with open(out_json_path, "r") as f:
                        f.close()
                    print("exception 3 fired at", datetime.now(),"on writing json_str to", \
                        out_json_path)
                    error_out = f"extract_one_note_pass2_10() caught error {err3=}, Type {type(err3)=}"
                    print(error_out)
            else:
                continue     # with next note

            if tot_note_cnt < len(file_list): 
                print("completed note", deid_note_key, "at", datetime.now(), \
                      "; waiting", sleep_secs, "seconds")
                time.sleep(sleep_secs)    # wait <sleep_secs> seconds to avoid timeout
            else: 
                print("completed note", deid_note_key, "at", datetime.now())
            tot_in_tokens += in_tokens
            tot_out_tokens += out_tokens
            tot_reason_tokens += reason_tokens
            with open(out_json_path, "a") as f: 
                f.write("\n")


    print("tot_in_tokens = ", tot_in_tokens, "; tot_out_tokens = ", tot_out_tokens, "; tot_reason_tokens = ",
         tot_reason_tokens)
    print("cost for model ", model_id, "; prompt= $"+ str(tot_in_tokens*price_per_in_token) \
                + "; completion= $" + str(tot_out_tokens*price_per_out_token) \
                + "; TOTAL = $" + str(tot_in_tokens*price_per_in_token \
                + tot_out_tokens*price_per_out_token) + "\n")
    print("program completed at", datetime.now())
    if len(err_note_key_list) > 0:
        print("Failed notes (if any):")
        print(err_note_key_list)
        for i in range(0, len(err_note_key_list), 5):
            err_note_string = "'" + "', '".join(err_note_key_list[i:i+5]) + \
                "' "+dir_delimiter
            print(err_note_key_list)

################# run pass 1 and then pass 2 #############################
# First pass 1
patientepicid_list = ""
out_dir = out_dir_prefix+dir_delimiter+out_dir_1_pass
in_note_path = in_dir_prefix + dir_delimiter + in_note_dir
system_text = system_text_pass1_v4
print("starting first pass")
query_and_extract_pass1_9(patientepicid_list, out_dir, system_text, \
   in_note_path, note_list = "", effort="medium", timeout=5600, \
   just_note_list=False, max_retries=2)

# Now pass 2
in_dir = out_dir_prefix+dir_delimiter+out_dir_1_pass
out_dir = out_dir_prefix+dir_delimiter+out_dir_2_pass
in_note_path = in_dir_prefix + dir_delimiter + in_note_dir
system_text = "system_text_pass2_v5"
# Execute the note extractions.
print("starting second pass")
query_and_extract_pass2_10(in_dir, out_dir, in_note_path, system_text, effort="medium", \
                       timeout=5600, max_retries=2)
print("finished second pass")
