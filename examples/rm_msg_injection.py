import json
from nltk.corpus import words
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel


def extract_response_dict_to_fb(message):
    # Find the start and end indices of the dictionary string
    start_index = message.find("{")
    end_index = message.find("}") + 1  # +1 to include the closing brace

    if start_index != -1 and end_index != -1 and start_index < end_index:
        dict_string = message[start_index:end_index]
        try:
            # Convert the string to a Python dictionary
            # json.loads expects keys to be strings, so we need a slight modification
            # for integer keys that look like they came from a Python dict literal.
            # ast.literal_eval is safer for Python literals.
            # import ast
            import json

            # result_dict = ast.literal_eval(dict_string)
            result_dict = eval(dict_string)
            return result_dict["response"].strip(), result_dict["rating"]

        except:
            return "", ""

    elif "READY" in message:
        return "READY", ""
    else:
        return message, ""


all_words_in_use = []


# self.trial_prompt,pred_dict,input_dict


def update_input_with_feedback(input_dict: dict, fb_response: str) -> dict:
    """Update the input dictionary with feedback response.

    Parameters:
    ----------
    input_dict : dict
        The input dictionary to be updated.
    fb_response : str
        The feedback response to be added.

    Returns:
    -------
    dict
        The updated input dictionary.
    """
    

    #input_msg = {"**feedba"}input_dict["inputs"][0]
    feedback_msg = fb_response
    feedback_msg = HumanMessage(feedback_msg)
    # if isinstance(input_msg.content, str):
    #    input_msg = json.loads(input_msg.content)
    #    input_msg["PREVIOUS TRIAL FEEDBACK"] = fb_response

    # input_dict["input_stim"][0] = HumanMessage(json.dumps(input_msg))
    
    trial_dict = json.loads(input_dict["inputs"][0].content) 
    updated_trial_with_fb = HumanMessage(json.dumps({
        **json.loads(fb_response),
        "current_trial": trial_dict},
        indent=4
        ))
    #print("---updated_trial---",updated_trial_with_fb)
    input_dict["inputs"] = [updated_trial_with_fb]
    return input_dict


def generate_fb_response(trdata, pred_dict, input_dict, parser_status, trial_item_collector=None):
    if trial_item_collector is None:
        trial_item_collector = []

    parser = parser_status
    response_fb = ""
    rate_fb = ""
    overall_fb = ""
    feedback_msg = ""
    message = pred_dict.content
    
    
    if parser == "1":
        message = eval(message)
        values = list(message.values())
        # Named lookup first so different parser schemas (2-field or 4-field) all work.
        # response: the word produced or the judgment made
        response = str(
            message.get("Word_2")
            or message.get("Judgment")
            or message.get("response")
            or (values[0] if values else "")
        )
        # rating: relatedness (0-100) for encode trials, confidence (1-6) for test trials
        rating = (
            message.get("Confidence")
            or message.get("Rating")
            or message.get("rating")
            or (values[1] if len(values) > 1 else "")
        )

    if parser == "0":
        response, rating = extract_response_dict_to_fb(message)
    
    
    stim = trdata["stimulus"]
    
    if "task_instruction" in trdata["trcode"]:
        if parser == "1":
            if "ready" not in response.lower():
                return "Correct answer is READY. Please carefully follow the instructions."
            else:
                return None
        else:
            if "ready" not in message.lower():
                return "Correct answer is READY. Please carefully follow the instructions."
            else:
                return None
    elif "test" in trdata["trcode"]:
        corr_ans = trdata["corrAns"].strip()
        if (
            response.lower().replace(".", "").strip()
            == corr_ans.lower().replace(".", "").strip()
        ):
            response_fb = f"**CORRECT: Correct generation for 'Word_2' response was {corr_ans}.  Your 'Word_2' response followed the task instructions**"
        else:
            response_fb = f"**INCORRECT: Correct generation for 'Word_2' response was {corr_ans}.**    It might be also be incorrect due to poor formatting of your ANSWER.    Carefully follow all the trial instructions about the 'response' options to be accurate on the given trial.    "
        try:
                rating = float(rating)
                if (rating >= 1) and (rating <= 6):
                    rate_fb = (
                        "**CORRECT: Your 'rating' value is in the instructed rating scale range.**"
                    )
                else:
                    rate_fb = "**INCORRECT: Your 'rating' value is INCORRECT as it is not in the instructed rating scale range.    It might be also be incorrect due to poor formatting of your ANSWER.**    Carefully follow all the trial instructions about the rating scale in the given trial instructions.    "

        except:
            rate_fb = "**INCORRECT: Your 'rating' value is INCORRECT due to problems in format of your answer.**    Carefully follow all the trial instructions about the rating scale in the given trial instructions.    "

    else:
        word1, word2 = (
            stim["Word_Pair"]["word_1"],
            stim["Word_Pair"]["word_2"],
        )  # word12.split(" and ")
        word1 = word1.strip()
        word2 = word2.strip()
        all_words_in_use.append(word1)
        if "__" in word2:
            if response.lower() in all_words_in_use:
                all_words_in_use.append(response)
                response_fb = "**INCORRECT: Last trial was a imagined trial type, your 'Word_2' value is INCORRECT because you imagined a second word to the incomplete word-pair in the trial which already have been used in the present trial as the first word or in any other way in previous trials. Always imagine the second word that is novel and have not been used in present and previous trials as your 'response' in a trial that is imagined trial type.    It might be also be incorrect due to poor formatting in your ANSWER.**    Carefully follow all the trial instructions related to imagined trials.    "

            elif response.lower() not in words.words():
                all_words_in_use.append(response)
                response_fb = "**INCORRECT: Last trial was a imagined trial type, your 'Word_2' value is INCORRECT because it is not in the English dictionary. Always imagine novel second word from english language to give as 'response' in trial similar to imagined trial type.    It might be also be incorrect due to poor formatting in your ANSWER.**    Carefully follow all the trial instructions related to imagined trials.    "

            else:
                response_fb = "**CORRECT: Last trial was a imagined trial type, your 'Word_2' correctly followed the given instructions.**    "
        else:
            all_words_in_use.append(word2)
            all_words_in_use.append(response)
            if word2.lower() != response.lower():
                response_fb = "**INCORRECT: Last trial was a perceived trial type, your 'Word_2' value is INCORRECT because the second word in the previous trial word-pair does not match your 'response'.    It might be also be incorrect due to poor formatting of your ANSWER.**    Carefully follow all the trial instructions related to perceived trials.    "
            else:
                response_fb = "**CORRECT: Last trial was a perceived trial type, your 'Word_2' value correctly followed the trial instruction for the given trial.**"
        try:
                rating = float(rating)
                if (rating >= 0) and (rating <= 100):
                    rate_fb = (
                        "**CORRECT: Your 'rating' value is in the instructed rating scale range.**"
                    )
                else:
                    rate_fb = "**INCORRECT: Your 'rating' value is INCORRECT as it is not in the instructed rating scale range.    It might be also be incorrect due to poor formatting of your ANSWER.**    Carefully follow all the trial instructions about the rating scale in the given trial.    "

        except:
                rate_fb = "**INCORRECT: Your 'rating' value is INCORRECT due to problems in format of your answer.**    Carefully follow all the trial instructions about the rating scale in the given trial.    "


    if any(["INCORRECT" in response_fb, "INCORRECT" in rate_fb]):
            overall_fb = "**INCORRECT Answer!** **Follow all the trial instructions more accurately to give correct response and rating.**"
    else:
        overall_fb = "**CORRECT Answer!** **Well Answered, Good Job!**"

    feedback_msg = json.dumps(
            {"Feedback on previous response": {
                "response feedback": response_fb,
                "rating feedback": rate_fb,
                "overall feedback": overall_fb,
            }},
            indent=4,
        )
    return feedback_msg

class Stim_Trial_Injection:
    def __init__(
        self,
        trial_data=None,
        pred_dict=None,
        check_input=None,
        parser_status=None,
        llmobj=None,
        trial_item_collector=None,
    ):
        self.trial_data = trial_data
        self.pred_dict = pred_dict
        self.check_input = check_input
        self.parser_status = parser_status
        self.llmobj = llmobj

        self.trialhash = {}
        self.trialhash["trcode"] = []
        self.trialhash["input"] = []
        self.trialhash["stim"] = []
        self.trialhash["output"] = []
        self.trialhash["response"] = []
        self.trialhash["rating"] = []

        self.all_stim_in_use = trial_item_collector

        self.fb_response = None

    def generate_feedback(
        self, trdata, pred_dict, input_dict, parser_status, trial_item_collector
    ):
        if trdata is None:
            trdata = self.trial_data
        else: 
            self.trial_data = trdata

        if pred_dict is None:
            pred_dict = self.pred_dict

        if input_dict is None:
            input_dict = self.check_input

        if parser_status is None:
            parser_status = self.parser_status

        if trial_item_collector is None:
            trial_item_collector = self.all_stim_in_use
        self.fb_response = generate_fb_response(
            trdata=trdata,
            pred_dict=pred_dict,
            input_dict=input_dict,
            parser_status=parser_status,
            trial_item_collector=trial_item_collector,
        )
        # print(self.fb_response,"xx")
        return self.fb_response

    def update_trial_stim(self, finput=None, fb_response=None, mod_finput=False):
        
        if finput is None:
            finput = self.trial_data["stimulus"]
        # if  fb_response is None:
        #    fb_response = self.generate_feedback()
        
        ffb_input = update_input_with_feedback(
            finput, fb_response
        )  # list of HumanMessage(content=<feedback>) and future input HumanMessage(content=<trial>)
        # if mod_finput:
        mfinput = ffb_input  # changes future input HumanMessage(content=<trial>)

        return mfinput
