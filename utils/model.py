import os
import openai
import pandas as pd
import time
import datetime
import re
from openai import OpenAI

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

class Model:
    def __init__(self, model_name, temperature=0.7, max_tokens=1024):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_name = model_name


    def get_model_name(self, model_name):
        return self.model_name


    def model_dump(self, user_prompt, system_prompt=""):
        # if system_prompt_path is not None: 
        #     with open(system_prompt_path, 'r', encoding='utf-8') as sys_file:
        #         system_prompt = sys_file.read()
        # if user_prompt_path is not None:
        #     raise ValueError("User prompt path cannot be None")
        # with open(user_prompt_path, 'r', encoding='utf-8') as user_file:
        #     user_prompt = user_file.read()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        logprobs = False
        top_logprobs = None
        try: 
            response = self.client.chat.completions.create(
                                model=self.model_name,
                                messages=messages,
                                logprobs=logprobs,
                                top_logprobs=top_logprobs,
                                temperature=self.temperature)
        except: 
            try: 
                response = self.client.chat.completions.create(
                                    model=self.model_name,
                                    messages=messages,
                                    logprobs=logprobs,
                                    top_logprobs=top_logprobs,
                                    temperature=self.temperature)
            except:
                return None
        
        return response.model_dump()



    def run_query(self, user_prompt, system_prompt=""):
        

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        logprobs = False
        top_logprobs = None
        try: 
            response = self.client.chat.completions.create(
                                model=self.model_name,
                                messages=messages,
                                logprobs=logprobs,
                                top_logprobs=top_logprobs,
                                temperature=self.temperature)
        except: 
            try: 
                response = self.client.chat.completions.create(
                                    model=self.model_name,
                                    messages=messages,
                                    logprobs=logprobs,
                                    top_logprobs=top_logprobs,
                                    temperature=self.temperature)
            except:
                return None
        
        return response.choices[0].message.content

# response = self.client.chat.completions.create(model=self.model_name, input=messages, logprobs=logprobs, top_logprobs=top_logprobs, temperature=self.temperature)