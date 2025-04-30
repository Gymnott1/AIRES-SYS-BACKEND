import requests
import json
import logging
from django.conf import settings
import re

logger = logging.getLogger(__name__)

def clean_json_string(json_str):
    
    print("\n===== CLEAN JSON STRING =====")
    print(f"Input JSON string length: {len(json_str)}")
    print(f"First 50 chars: {json_str[:50]}...")
    
    cleaned = json_str.strip()
    print(f"After initial strip, length: {len(cleaned)}")

    print("Removing code fence markers if present...")
    cleaned = cleaned.strip('```json')
    cleaned = cleaned.strip('```')
    cleaned = cleaned.strip() 
    print(f"After removing code fences, length: {len(cleaned)}")

    print("Replacing escaped newlines...")
    cleaned = cleaned.replace('\\\n', '\n')
    print(f"After newline replacement, length: {len(cleaned)}")

    print("Removing control characters...")
    cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', cleaned)
    print(f"After control char removal, length: {len(cleaned)}")
   
    print("Fixing trailing commas...")
    cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
    print(f"After trailing comma fix, length: {len(cleaned)}")

    print(f"Cleaned JSON string: {cleaned[:100]}...")
    logger.debug(f"Cleaned JSON string attempt: {cleaned[:100]}...") 
    print("===== CLEAN JSON STRING COMPLETE =====")
    return cleaned

def query_llm(prompt, task_type="generation", parameters=None):
    
    print("\n===== QUERY LLM =====")
    print(f"Task type: {task_type}")
    print(f"Prompt length: {len(prompt)}")
    print(f"Prompt preview: {prompt[:100]}...")
    print(f"Input parameters: {parameters}")
    
    if parameters is None:
        print("No parameters provided, using defaults")
        parameters = {}

    
    print("Setting up default parameters...")
    default_params = {
        'max_tokens': 500,  
        'temperature': 0.7,
        'do_sample': True,
        'return_full_text': False,
        'ollama_options': {
            "num_predict": parameters.get('max_tokens', 500),  
            "temperature": parameters.get('temperature', 0.7),
            "top_p": parameters.get('top_p', 0.9),
            "top_k": parameters.get('top_k', 40),
            "stop": parameters.get('stop', []),
        }
    }
    print(f"Default parameters: {default_params}")

    if task_type == "classification":
        print("Task is classification, adjusting parameters...")
        default_params['max_tokens'] = 300  
        default_params['ollama_options']['num_predict'] = 300
        print(f"Adjusted default parameters: {default_params}")

    print("Merging provided parameters with defaults...")
    final_params = default_params.copy()
    final_params.update(parameters)
    print(f"Final merged parameters: {final_params}")
    
    print("Updating Ollama specific options...")
    final_params['ollama_options']['num_predict'] = final_params.get('max_new_tokens', 
                                                  final_params.get('max_tokens', 500))
    final_params['ollama_options']['temperature'] = final_params.get('temperature', 0.7)
    print(f"Updated Ollama options: {final_params['ollama_options']}")

    if len(prompt) > 1500 and task_type == "classification":
        print("WARNING: Prompt exceeds 1500 characters for classification, trimming")
        logger.warning("Trimming prompt for classification as it exceeds 1500 characters")
        prompt = prompt[:1500]
        print(f"Trimmed prompt length: {len(prompt)}")

    print("Selecting appropriate model based on task type...")
    local_model = settings.LOCAL_LLM_MODEL_NAME
    print(f"Initial model selection: {local_model}")
    
    if task_type == "sentiment":
        if hasattr(settings, 'LOCAL_SENTIMENT_MODEL_NAME'):
            local_model = settings.LOCAL_SENTIMENT_MODEL_NAME
            print(f"Using sentiment-specific model: {local_model}")
    elif task_type == "classification":
        if hasattr(settings, 'LOCAL_CLASSIFICATION_MODEL_NAME'):
            local_model = settings.LOCAL_CLASSIFICATION_MODEL_NAME
            print(f"Using classification-specific model: {local_model}")
    
    print(f"Final model selection: {local_model}")

    generated_text = None
    response_json = None
    error_info = None

    if settings.USE_LOCAL_LLM:
        print(f"\n----- ATTEMPTING LOCAL LLM QUERY -----")
        print(f"Local LLM API URL: {settings.LOCAL_LLM_API_URL}")
        print(f"Local model: {local_model}")
        try:
            logger.info(f"Attempting local LLM query to {settings.LOCAL_LLM_API_URL} with model {local_model}")
            
            formatted_prompt = prompt
            if task_type == "classification":
                print("Simplifying prompt for classification task...")
                
                formatted_prompt = f"Classify this text into one category: {prompt[:800]}"
                if 'candidate_labels' in parameters:
                    formatted_prompt += f"\nCategories: {', '.join(parameters['candidate_labels'])}"
                print(f"Simplified prompt length: {len(formatted_prompt)}")
                print(f"Simplified prompt preview: {formatted_prompt[:100]}...")
            
            print("Preparing payload for local LLM...")
            payload = {
                "model": local_model,
                "prompt": formatted_prompt,
                "stream": False,
                "options": final_params['ollama_options']
            }
            print(f"Payload prepared with prompt length: {len(payload['prompt'])}")
            print(f"Payload options: {payload['options']}")
            
            print(f"Making POST request to {settings.LOCAL_LLM_API_URL}...")
            print(f"Timeout setting: {settings.LOCAL_LLM_TIMEOUT + 30}s")
            response = requests.post(
                settings.LOCAL_LLM_API_URL,
                json=payload,
                timeout=settings.LOCAL_LLM_TIMEOUT + 30, 
                headers={"Content-Type": "application/json"}
            )
            
            print(f"Response received with status code: {response.status_code}")
            response.raise_for_status()  
            response_json = response.json()
            print(f"Response JSON keys: {response_json.keys()}")

            if 'response' in response_json:
                generated_text = response_json['response']
                print(f"Generated text retrieved, length: {len(generated_text)}")
                print(f"Preview: {generated_text[:100]}...")
                logger.info("Local LLM query successful.")
                
                if task_type == 'generation':
                    print("Task is generation, returning generated text")
                    print("----- LOCAL LLM QUERY SUCCESSFUL -----")
                    print("===== QUERY LLM COMPLETE =====")
                    return {"generated_text": generated_text}
                elif task_type in ['classification', 'sentiment']:
                    print(f"Task is {task_type}, attempting to parse JSON response...")
                   
                    try:
                        
                        print("Cleaning response text for JSON parsing...")
                        clean_text = clean_json_string(generated_text)
                        
                        if not clean_text.strip().startswith('{'):
                            print("Adding missing opening brace")
                            clean_text = '{' + clean_text
                        
                        if not clean_text.strip().endswith('}'):
                            print("Adding missing closing brace")
                            clean_text = clean_text + '}'
                        
                        print(f"Final JSON string to parse: {clean_text[:100]}...")
                        print("Attempting to parse JSON...")
                        result = json.loads(clean_text)
                        print(f"JSON successfully parsed with keys: {result.keys()}")
                        
                        if task_type == 'sentiment' and 'sentiment' in result:
                            print(f"Sentiment response found with value: {result['sentiment']}")
                            print("----- LOCAL LLM QUERY SUCCESSFUL -----")
                            print("===== QUERY LLM COMPLETE =====")
                            return result
                        elif task_type == 'classification' and ('labels' in result or 'classifications' in result):
                            
                            if 'classifications' in result and not 'labels' in result:
                                print("Standardizing classification response format...")
                                result['labels'] = [c['label'] for c in result['classifications']]
                                result['scores'] = [c['score'] for c in result['classifications']]
                                print(f"Standardized result: {result}")
                            
                            print("----- LOCAL LLM QUERY SUCCESSFUL -----")
                            print("===== QUERY LLM COMPLETE =====")
                            return result
                        else:
                            
                            print("Unexpected but valid JSON structure, returning as general result")
                            print("----- LOCAL LLM QUERY SUCCESSFUL -----")
                            print("===== QUERY LLM COMPLETE =====")
                            return {"result": result, "raw_response": generated_text}
                    except json.JSONDecodeError as e:
                        print(f"ERROR: Failed to parse response as JSON: {e}")
                        print(f"Problematic JSON: {clean_text[:100]}...")
                        logger.warning(f"Failed to parse LLM response as JSON: {e}")
                        
                        print("----- LOCAL LLM QUERY COMPLETED WITH PARSING ERROR -----")
                        print("===== QUERY LLM COMPLETE =====")
                        return {"raw_response": generated_text, 
                                "error": "Failed to parse as JSON",
                                "message": "The model returned text that could not be parsed as valid JSON."}
            elif 'error' in response_json:
                error_info = {"error": "Local LLM API Error", "details": response_json['error']}
                print(f"ERROR: Local LLM API returned an error: {response_json['error']}")
                logger.error(f"Local LLM API returned an error: {response_json['error']}")
            else:
                error_info = {"error": "Unexpected local LLM response format", "details": response_json}
                print(f"ERROR: Unexpected local LLM response format: {response_json}")
                logger.error(f"Unexpected local LLM response format: {response_json}")

        except requests.exceptions.RequestException as e:
            error_info = {"error": "Local LLM connection failed", "details": str(e)}
            print(f"ERROR: Local LLM connection failed: {e}")
            print(f"Exception type: {type(e)}")
            logger.error(f"Local LLM connection error: {e}")
        except Exception as e:
            error_info = {"error": "Error during local LLM processing", "details": str(e)}
            print(f"ERROR: Unexpected error during local LLM processing: {e}")
            print(f"Exception type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            logger.exception("Unexpected error during local LLM processing:") 
        
        print(f"----- LOCAL LLM QUERY {'FAILED' if error_info else 'DISABLED'} -----")

    else:
        print("Local LLM is disabled in settings")

    if generated_text is None:  
        print("\n----- ATTEMPTING HUGGING FACE API FALLBACK -----")
        if not settings.HF_API_KEY:
            print("ERROR: HF_API_KEY not configured in settings")
            logger.error("Fallback to HF failed: HF_API_KEY not configured.")
            print("----- HUGGING FACE API FALLBACK FAILED -----")
            print("===== QUERY LLM FAILED =====")
            return error_info or {"error": "LLM unavailable", "details": "Local failed and HF API key missing."}

        print("Selecting appropriate HF model endpoint...")
        if task_type == "sentiment":
            fallback_url = getattr(settings, 'HF_SENTIMENT_MODEL_URL', settings.HF_CLASSIFICATION_MODEL_URL)
            print(f"Using sentiment endpoint: {fallback_url}")
        else:
            fallback_url = settings.HF_GENERATION_MODEL_URL if task_type == "generation" else settings.HF_CLASSIFICATION_MODEL_URL
            print(f"Using {'generation' if task_type == 'generation' else 'classification'} endpoint: {fallback_url}")
        
        headers = {
            "Authorization": f"Bearer {settings.HF_API_KEY[:5]}...",  
            "Content-Type": "application/json"
        }
        print(f"Headers prepared: {headers}")
        
        print("Preparing HF API payload...")
        hf_payload = {
            "inputs": prompt,
           
            "parameters": {
                k: v for k, v in final_params.items() if k not in ['ollama_options', 'max_tokens']  
            }
        }
        print(f"Initial HF payload parameters: {hf_payload['parameters']}")
        
        if 'max_new_tokens' in final_params:
            print(f"Setting max_new_tokens to {final_params['max_new_tokens']}")
            hf_payload['parameters']['max_new_tokens'] = final_params['max_new_tokens']
        elif 'max_tokens' in final_params:
            print(f"Converting max_tokens to max_new_tokens: {final_params['max_tokens']}")
            hf_payload['parameters']['max_new_tokens'] = final_params['max_tokens']

        if task_type in ['classification', 'sentiment']:
            print(f"Adding {task_type}-specific parameters...")
            if task_type == 'classification':
                
                candidate_labels = parameters.get("candidate_labels", [
                    "resume", "curriculum vitae", "CV", "job application",
                    "article", "report", "manual", "academic paper", "letter",
                    "other"
                ])
                print(f"Setting classification candidate labels: {candidate_labels}")
                hf_payload['parameters']['candidate_labels'] = candidate_labels
            elif task_type == 'sentiment':
                
                candidate_labels = parameters.get("candidate_labels", [
                    "positive", "negative", "neutral"
                ])
                print(f"Setting sentiment candidate labels: {candidate_labels}")
                hf_payload['parameters']['candidate_labels'] = candidate_labels
            
            if 'do_sample' in hf_payload['parameters']:
                print("Removing do_sample parameter for classification/sentiment")
                hf_payload['parameters'].pop('do_sample', None)
            
            if 'temperature' in hf_payload['parameters']:
                print("Removing temperature parameter for classification/sentiment")
                hf_payload['parameters'].pop('temperature', None)

        print(f"Final HF payload: inputs length={len(hf_payload['inputs'])}, parameters={hf_payload['parameters']}")
        logger.warning(f"Local LLM failed or disabled. Falling back to Hugging Face API: {fallback_url}")
        
        try:
            print(f"Making POST request to HF API at {fallback_url}...")
            print(f"Timeout setting: {settings.HF_API_TIMEOUT}s")
            hf_response = requests.post(
                fallback_url,
                headers=headers,
                json=hf_payload,
                timeout=settings.HF_API_TIMEOUT
            )
            print(f"Response received with status code: {hf_response.status_code}")
            hf_response.raise_for_status()
            result = hf_response.json()
            print(f"Response parsed as JSON: {type(result)}")
            if isinstance(result, list):
                print(f"List response with {len(result)} items")
            elif isinstance(result, dict):
                print(f"Dict response with keys: {result.keys()}")
            else:
                print(f"Unexpected response type: {type(result)}")
            
            logger.info("Hugging Face API fallback successful.")

            if isinstance(result, list) and len(result) > 0 and task_type == "generation":
                print("Unwrapping list response for generation task")
                
                print("----- HUGGING FACE API FALLBACK SUCCESSFUL -----")
                print("===== QUERY LLM COMPLETE =====")
                return result[0]  
            elif isinstance(result, dict):  
                print("Returning dictionary response directly")
                print("----- HUGGING FACE API FALLBACK SUCCESSFUL -----")
                print("===== QUERY LLM COMPLETE =====")
                return result
            else:  
                print(f"ERROR: Unexpected HF API response format: {result}")
                logger.error(f"Unexpected HF API response format: {result}")
                print("----- HUGGING FACE API FALLBACK FAILED -----")
                print("===== QUERY LLM FAILED =====")
                return {"error": "Unexpected HF API response format", "details": result}

        except requests.exceptions.RequestException as e:
            print(f"ERROR: HF API request failed: {e}")
            print(f"Exception type: {type(e)}")
            logger.error(f"Hugging Face API fallback failed: {e}")
            
            final_error = error_info or {"error": "LLM unavailable"}
            final_error["fallback_error"] = {"error": "HF API connection failed", "details": str(e)}
            print("----- HUGGING FACE API FALLBACK FAILED -----")
            print("===== QUERY LLM FAILED =====")
            return final_error
        except Exception as e:
            print(f"ERROR: Unexpected error during HF API fallback: {e}")
            print(f"Exception type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            logger.exception("Unexpected error during HF API fallback:")
            final_error = error_info or {"error": "LLM unavailable"}
            final_error["fallback_error"] = {"error": "Error during HF API processing", "details": str(e)}
            print("----- HUGGING FACE API FALLBACK FAILED -----")
            print("===== QUERY LLM FAILED =====")
            return final_error

    print("WARNING: Reached end of function without returning, this should not happen")
    print("===== QUERY LLM FAILED =====")
    return error_info or {"error": "LLM query failed"}



def analyze_sentiment(text, parameters=None):
    
    print("\n===== ANALYZE SENTIMENT =====")
    print(f"Text length: {len(text)}")
    print(f"Text preview: {text[:100]}...")
    print(f"Parameters: {parameters}")
    
    result = query_llm(text, task_type="sentiment", parameters=parameters)
    
    print(f"Sentiment analysis result type: {type(result)}")
    if isinstance(result, dict):
        if 'error' in result:
            print(f"ERROR in sentiment analysis: {result['error']}")
        elif 'sentiment' in result:
            print(f"Sentiment detected: {result['sentiment']}")
        else:
            print(f"Result keys: {result.keys()}")
    
    print("===== ANALYZE SENTIMENT COMPLETE =====")
    return result