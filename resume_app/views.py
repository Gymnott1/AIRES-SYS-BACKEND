import os, io
from rest_framework.views import APIView
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework import status
from .models import Resume, ChatMessage
from .serializers import ResumeSerializer, ChatMessageSerializer
import PyPDF2
import requests
import json
from django.http import JsonResponse
from django.conf import settings
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from rest_framework.authtoken.models import Token
from django.contrib.auth import update_session_auth_hash
from django.http import FileResponse
from django.http import HttpResponse
from reportlab.lib.pagesizes import letter
import re
from django.views.decorators.clickjacking import xframe_options_exempt
from django.utils.decorators import method_decorator
from .models import Resume
from mistralai import Mistral, UserMessage, SystemMessage
from rest_framework.decorators import api_view, permission_classes
from PyPDF2 import PdfReader, errors as PyPDF2Errors
import traceback





HF_API_KEY = "ENTER_YOUR_HUGGINGFACE_API_KEY_HERE"


GITHUB_TOKEN=""


try:
    from PyPDF2 import PdfReader, errors as PyPDF2Errors
except ImportError:
    # Handle case where PyPDF2 might not be installed or has issues
    PdfReader = None
    PyPDF2Errors = None
    print("WARNING: PyPDF2 library not found or failed to import. PDF processing will fail.")

try:
    from mistralai import Mistral, UserMessage, SystemMessage
except ImportError:
    Mistral = None
    UserMessage = None
    SystemMessage = None
    print("WARNING: mistralai library not found. AI analysis will fail.")

from .models import Resume # Assuming you might want to link later # Use environment variable
MAX_RECRUITER_FILES = 5 # Match frontend setting

class RecruiterAnalyzeView(APIView):
    permission_classes = [AllowAny] # Adjust as needed

    def extract_text_from_pdf(self, pdf_file_obj):
        """Extracts text from an InMemoryUploadedFile or similar file object."""
        if PdfReader is None:
             raise ImportError("PyPDF2 library is required for PDF processing.")

        text = ""
        try:
            pdf_file_obj.seek(0)
            pdf_reader = PdfReader(pdf_file_obj)
            if not pdf_reader.pages:
                 print(f"Warning: PDF file {getattr(pdf_file_obj, 'name', 'N/A')} has no pages or is unreadable.")
                 return None # Indicate failure

            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            if not text.strip():
                 print(f"Warning: Could not extract any text from PDF {getattr(pdf_file_obj, 'name', 'N/A')}.")
                 return None # Indicate failure

            return text
        except PyPDF2Errors.PdfReadError as pdf_err:
            print(f"Error reading PDF {getattr(pdf_file_obj, 'name', 'N/A')}: {pdf_err}")
            if "encrypted" in str(pdf_err).lower():
                raise ValueError(f"File '{getattr(pdf_file_obj, 'name', 'N/A')}' is encrypted and cannot be processed.")
            else:
                raise ValueError(f"File '{getattr(pdf_file_obj, 'name', 'N/A')}' is corrupted or not a valid PDF.")
        except Exception as e:
            print(f"Unexpected error extracting text from {getattr(pdf_file_obj, 'name', 'N/A')}: {e}")
            traceback.print_exc()
            raise ValueError(f"Failed to process file '{getattr(pdf_file_obj, 'name', 'N/A')}'.")

    def clean_ai_json_response(self, raw_text):
        """Attempts to extract a valid JSON object from the AI's raw text output."""
        cleaned = re.sub(r'```json\s*', '', raw_text, flags=re.IGNORECASE)
        cleaned = re.sub(r'```', '', cleaned)
        cleaned = cleaned.strip()

        first_brace = cleaned.find('{')
        last_brace = cleaned.rfind('}')

        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_str = cleaned[first_brace : last_brace + 1]
            try:
                # Attempt to parse the extracted string
                # Replace escaped newlines potentially missed by AI
                json_str_corrected = json_str.replace('\\n', '\n')
                # Try parsing with strict=False first if needed, but aim for valid JSON
                parsed_json = json.loads(json_str_corrected)
                return parsed_json
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error after extraction: {e}")
                print(f"Problematic JSON string segment (first 500 chars):\n {json_str[:500]}...")
                # Try to fix common issues like trailing commas (requires more advanced parsing)
                # For now, return None to indicate failure
                return None
        else:
            print("Could not find valid start/end braces for JSON object.")
            return None

    def post(self, request, format=None):
        if Mistral is None:
             return Response({'error': 'AI analysis service is not configured correctly on the server.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        job_description = request.data.get('job_description')
        resume_files = request.FILES.getlist('resumes')

        # --- Validation ---
        if not job_description or not job_description.strip():
            return Response({'error': 'Job description is required.'}, status=status.HTTP_400_BAD_REQUEST)
        if not resume_files:
            return Response({'error': 'At least one resume file (PDF) is required.'}, status=status.HTTP_400_BAD_REQUEST)
        # --- Use MAX_RECRUITER_FILES ---
        if len(resume_files) > MAX_RECRUITER_FILES:
            return Response({'error': f'A maximum of {MAX_RECRUITER_FILES} resume files can be uploaded.'}, status=status.HTTP_400_BAD_REQUEST)

        # --- Process Resumes ---
        extracted_texts = []
        file_identifiers = []
        processing_errors = [] # Collect non-fatal errors
        for index, file_obj in enumerate(resume_files):
            if not file_obj.name.lower().endswith('.pdf'):
                 processing_errors.append(f"File '{file_obj.name}' is not a PDF and was skipped.")
                 continue # Skip non-PDF files
            try:
                text = self.extract_text_from_pdf(file_obj)
                if text is None:
                    processing_errors.append(f"Could not extract text from '{file_obj.name}' (might be image-based or corrupted); it was skipped.")
                    continue # Skip files where text extraction failed
                extracted_texts.append(text)
                file_identifiers.append(file_obj.name)
            except ValueError as extraction_error:
                # Catch critical errors raised from extract_text_from_pdf
                return Response({'error': str(extraction_error)}, status=status.HTTP_400_BAD_REQUEST)
            except ImportError as import_err:
                 # Raised if PyPDF2 wasn't loaded
                 return Response({'error': 'PDF processing library is unavailable on the server.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            except Exception as e:
                 print(f"Unexpected error processing file {file_obj.name}: {e}")
                 traceback.print_exc()
                 return Response({'error': f"An unexpected server error occurred while processing '{file_obj.name}'."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Check if any resumes were successfully processed
        if not extracted_texts:
             error_detail = " ".join(processing_errors) if processing_errors else "No valid PDF resumes were found or could be processed."
             return Response({'error': 'No resumes could be processed.', 'details': error_detail}, status=status.HTTP_400_BAD_REQUEST)

        # --- Construct ENHANCED AI Prompt ---
        resume_sections = ""
        for i, text in enumerate(extracted_texts):
            # Use the stored identifier corresponding to the successfully extracted text
            resume_sections += f"--- Resume {i+1} (Identifier: {file_identifiers[i]}) ---\n"
            resume_sections += f"{text}\n\n" # Added double newline for clarity

        # *** THIS IS THE UPDATED PROMPT SECTION ***
        prompt = f"""
You are an expert AI hiring assistant performing a detailed comparison of multiple candidate resumes against a specific job description.

Job Description:
---
{job_description}
---

Resumes:
{resume_sections}---

Task:
Analyze each resume thoroughly against the job description and provide a structured JSON response.

Instructions:
1.  **Overall Evaluation:** For EACH resume, provide an overall `match_score` (integer 0-100) representing alignment with the **entire** job description.
2.  **Categorical Scores:** For EACH resume, provide specific scores (integer 0-100) for:
    *   `skills_score`: How well the candidate's skills match the required/desired skills.
    *   `experience_score`: Relevance and depth of work experience compared to job requirements.
    *   `education_score`: Alignment of educational background with job requirements.
    *   `keyword_score`: Presence and relevance of keywords from the job description within the resume.
3.  **Keyword Analysis:** For EACH resume:
    *   Identify and list the top 5-10 most important `keywords_matched` from the job description found in the resume.
    *   Identify and list the top 5-10 most important `keywords_missing` from the job description *not* found in the resume.
4.  **Strengths & Weaknesses:** For EACH resume, provide concise lists (3-5 bullet points each) *relative to this specific job description*:
    *   `strengths`: Key qualifications, experiences, or skills that make the candidate a good fit.
    *   `weaknesses`: Areas where the candidate falls short of the job requirements or potential gaps.
5.  **Red Flags:** For EACH resume, list any potential `red_flags` (0-3 bullet points). Examples: unexplained employment gaps, frequent short-term jobs, lack of specific core requirements, poor formatting impacting readability. If none, provide an empty list [].
6.  **Recommendation Tier:** For EACH resume, assign a `recommendation_tier` from the following options: "Top Match", "Strong Candidate", "Potential Fit", "Less Suitable". Base this on the overall analysis.
7.  **AI Summaries:**
    *   Provide a brief `job_description_summary` (1-2 sentences) capturing the essence of the role.
    *   Provide a concise `summary` (2-3 sentences) for EACH candidate, highlighting their overall suitability *for this role*.
8.  **Comparative Analysis:** Provide a `comparative_analysis` section (1-2 paragraphs) comparing the candidates head-to-head *for this specific role*, highlighting key differentiators and trade-offs.
9.  **Ranking:** Provide a `ranking` list containing the exact `resume_identifier` strings (provided in the input) ordered from most suitable to least suitable based on your comprehensive analysis.
10. **Output Format:** Respond ONLY with a single, valid JSON object adhering strictly to the structure below. Ensure all strings are properly escaped. Do not include any text, explanations, or markdown code fences outside the JSON object.

Expected JSON Structure:
{{
  "job_description_summary": "string",
  "comparative_analysis": "string",
  "ranking": [
    "string (resume_identifier 1)",
    "string (resume_identifier 2)",
    // ... include all provided identifiers in order
  ],
  "candidate_analysis": [
    {{
      "resume_identifier": "{file_identifiers[0]}", // Use the actual identifier
      "match_score": integer (0-100),
      "recommendation_tier": "string ('Top Match' | 'Strong Candidate' | 'Potential Fit' | 'Less Suitable')",
      "summary": "string (2-3 sentences summary vs JD)",
      "details": {{
        "skills_score": integer (0-100),
        "experience_score": integer (0-100),
        "education_score": integer (0-100),
        "keyword_score": integer (0-100),
        "keywords_matched": ["string", "string", ...],
        "keywords_missing": ["string", "string", ...],
        "strengths": ["string (bullet point)", "string", ...],
        "weaknesses": ["string (bullet point)", "string", ...],
        "red_flags": ["string (bullet point)", "string", ...] // Empty list [] if none
      }}
    }}
    // Repeat this structure for each processed resume, using its correct identifier
    // e.g., {{ "resume_identifier": "{file_identifiers[1]}", ... }}
  ]
}}

Your JSON Response:
"""
        # *** END OF UPDATED PROMPT SECTION ***


        # --- Call AI Service ---
        token = GITHUB_TOKEN # Ensure this is correctly loaded
        endpoint = "https://models.inference.ai.azure.com" # Verify endpoint
        model_name = "mistral-small-2503" # Or your preferred model

        if not token or token == "YOUR_MISTRAL_API_KEY_FALLBACK":
             print("ERROR: GITHUB_TOKEN (Mistral API Key) is not set correctly.")
             # Avoid exposing key details in the error response
             return Response({'error': 'AI service configuration error on the server.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        client = Mistral(api_key=token, server_url=endpoint)
        analysis_json = None

        try:
            print(f"DEBUG: Sending request to AI for recruiter analysis ({len(extracted_texts)} resumes)...")
            response = client.chat.complete(
                model=model_name,
                messages=[
                    SystemMessage(content="You are an expert AI hiring assistant comparing resumes to job descriptions accurately and objectively."),
                    UserMessage(content=prompt),
                ],
                temperature=0.4, # Slightly lower for more consistent structured output
                max_tokens=4000, # Increased slightly due to more detailed output per candidate
                top_p=1.0
                # Consider adding response_format={"type": "json_object"} if using newer models/APIs that support it
            )
            print("DEBUG: Received AI response.")

            raw_ai_text = response.choices[0].message.content
            # Log less in production maybe, or only on error
            # print(f"DEBUG: Raw AI response text:\n---\n{raw_ai_text[:1000]}...\n---")

            analysis_json = self.clean_ai_json_response(raw_ai_text)

            if analysis_json is None:
                print("ERROR: Failed to parse JSON from AI response after cleaning.")
                # Provide a more helpful error message if parsing failed
                return Response({
                    'error': 'The AI response could not be processed into the expected format.',
                    'details': 'The analysis service returned data that was not valid JSON. Please try again. If the problem persists, contact support.'
                    # Optionally include a non-sensitive part of the raw response for debugging if safe
                    # 'raw_snippet': raw_ai_text[:200] + '...'
                 }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # --- ENHANCED Validation of Parsed JSON ---
            required_top_level = ['job_description_summary', 'comparative_analysis', 'ranking', 'candidate_analysis']
            if not all(key in analysis_json for key in required_top_level):
                 missing_keys = [key for key in required_top_level if key not in analysis_json]
                 print(f"ERROR: Parsed JSON missing top-level keys: {missing_keys}")
                 return Response({'error': f'AI analysis is incomplete. Missing top-level sections: {", ".join(missing_keys)}.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            if not isinstance(analysis_json.get('candidate_analysis'), list) or not analysis_json['candidate_analysis']:
                 print("ERROR: 'candidate_analysis' is not a list or is empty.")
                 return Response({'error': 'AI analysis is incomplete. No candidate analysis data found.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Validate the structure of the first candidate's analysis as a sample
            first_candidate = analysis_json['candidate_analysis'][0]
            required_candidate_keys = ['resume_identifier', 'match_score', 'recommendation_tier', 'summary', 'details']
            if not all(key in first_candidate for key in required_candidate_keys):
                 missing_keys = [key for key in required_candidate_keys if key not in first_candidate]
                 print(f"ERROR: Parsed JSON missing required candidate keys: {missing_keys}")
                 return Response({'error': f'AI analysis is incomplete. Missing candidate details: {", ".join(missing_keys)}.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            required_details_keys = ['skills_score', 'experience_score', 'education_score', 'keyword_score', 'keywords_matched', 'keywords_missing', 'strengths', 'weaknesses', 'red_flags']
            if not isinstance(first_candidate.get('details'), dict) or not all(key in first_candidate['details'] for key in required_details_keys):
                 missing_keys = [key for key in required_details_keys if not isinstance(first_candidate.get('details'), dict) or key not in first_candidate['details']]
                 print(f"ERROR: Parsed JSON missing required candidate 'details' keys: {missing_keys}")
                 return Response({'error': f'AI analysis is incomplete. Missing granular candidate details: {", ".join(missing_keys)}.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Basic type checks (optional but good practice)
            if not isinstance(first_candidate['match_score'], int) or not isinstance(first_candidate['details']['skills_score'], int):
                 print("ERROR: Score fields are not integers.")
                 # return Response({'error': 'AI analysis format error: Scores must be integers.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                 # Be lenient for now, frontend might handle string scores if needed
                 pass
            if not isinstance(first_candidate['details']['keywords_matched'], list) or not isinstance(first_candidate['details']['strengths'], list):
                 print("ERROR: Keyword/Strength/Weakness/RedFlag fields are not lists.")
                 return Response({'error': 'AI analysis format error: Expected lists for keywords, strengths, etc.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            print("DEBUG: Successfully parsed and validated AI analysis JSON structure.")

            # --- Include Processing Errors in Response (Optional) ---
            if processing_errors:
                analysis_json['processing_warnings'] = processing_errors

            return Response(analysis_json, status=status.HTTP_200_OK)

        except requests.exceptions.RequestException as api_error:
            print(f"Error calling AI API: {api_error}")
            # Log more details for server logs, less for client
            error_details_for_log = f"Status: {getattr(api_error.response, 'status_code', 'N/A')}, Body: {getattr(api_error.response, 'text', 'N/A')[:500]}..." if hasattr(api_error, 'response') and api_error.response is not None else str(api_error)
            print(f"AI API Request Error Details: {error_details_for_log}")
            return Response({'error': 'Could not connect to the AI analysis service. Please try again later.'}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        except Exception as e:
            # Catch-all for other unexpected errors during AI call or processing
            print(f"Unexpected error during AI processing or response handling: {e}")
            traceback.print_exc()
            return Response({'error': 'An unexpected error occurred during analysis.', 'details': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)




# @method_decorator(xframe_options_exempt, name='dispatch')
class ResumePDFView(APIView):
    permission_classes = [AllowAny]  

    def get(self, request, pk):
        try:
            resume = Resume.objects.get(pk=pk)
            
            if not resume.file or not resume.file.name:
                 return Response(
                    {"error": "Resume record exists but has no associated file."},
                    status=status.HTTP_404_NOT_FOUND
                )

            file_name = os.path.basename(resume.file.name)
            
            if hasattr(resume.file, 'path') and resume.file.path:
                 file_path = resume.file.path
            else:
                 
                 file_path = os.path.join(settings.MEDIA_ROOT, 'resumes', file_name)


            print(f"DEBUG: Attempting to access PDF at path: {file_path}") # Add debug print

            if not os.path.exists(file_path):
                print(f"ERROR: Resume file not found at path: {file_path}") 
                alt_path = os.path.join(settings.MEDIA_ROOT, file_name)
                if os.path.exists(alt_path):
                     print(f"DEBUG: Found file at alternate path: {alt_path}")
                     file_path = alt_path
                else:
                     print(f"ERROR: Also not found at alternate path: {alt_path}")
                     return Response(
                        {"error": f"Resume file '{file_name}' not found."},
                        status=status.HTTP_404_NOT_FOUND
                     )

            
            file_handle = open(file_path, 'rb')
            response = FileResponse(file_handle, content_type='application/pdf')
            
            response['X-Frame-Options'] = 'SAMEORIGIN'
            
            return response
        except Resume.DoesNotExist:
            return Response(
                {"error": "Resume not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            print(f"ERROR retrieving PDF: {e}") 
            import traceback
            traceback.print_exc() 
            return Response(
                {"error": "Error retrieving PDF", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )




def clean_json_string(json_str):
    
    cleaned = json_str.replace('\\\n', '\n')
    
    cleaned = re.sub(r'[\x00-\x1F\x7F]', '', cleaned)
    return cleaned

class ResumeValidator:
    def __init__(self):
        self.api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
        self.headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    
    def extract_text(self, pdf_file):
        
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    
    def is_resume_local(self, text):
        """Use a simple keyword-based approach to determine if text is likely a resume"""
        
        text_lower = text.lower()
        
        resume_indicators = [
            "work experience", "employment history", "professional experience",
            "education", "skills", "certifications", "references",
            "objective", "summary", "professional summary", "career objective",
            "technical skills", "work history", "job history",
            "achievements", "accomplishments", "projects",
            "languages", "proficient in", "expertise in"
        ]
        
        
        indicator_count = sum(1 for indicator in resume_indicators if indicator in text_lower)
        
        has_email = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        has_phone = bool(re.search(r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', text))
        has_linkedin = "linkedin.com" in text_lower
        
        personal_info_count = sum([has_email, has_phone, has_linkedin])
        
        max_possible_indicators = len(resume_indicators) + 3  
        confidence = min(1.0, (indicator_count + personal_info_count) / (max_possible_indicators * 0.5))
        
        is_valid = confidence > 0.3
        
        return is_valid, {
            "is_resume": is_valid,
            "confidence": confidence,
            "indicators_found": indicator_count,
            "personal_info_found": personal_info_count,
            "top_label": "resume" if is_valid else "other",
            "service_used": "local_keyword_analysis"
        }

    def is_resume_hf(self, text):
        """Use Hugging Face's API to determine if the text is a resume"""
        try:
            
            payload = {
                "inputs": text[:1024],  
                "parameters": {
                    "candidate_labels": [
                        "resume", "curriculum vitae", "CV", "job application",
                        "article", "report", "manual", "academic paper", "letter",
                        "other"
                    ]
                }
            }
           
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            result = response.json()
            print(result)
            
            resume_labels = ["resume", "curriculum vitae", "CV", "job application", "academic paper"]
            
            if "labels" in result and "scores" in result:
                top_label = result["labels"][0]
                top_score = result["scores"][0]
                
                is_valid = top_label in resume_labels
                
                return is_valid, {
                    "is_resume": is_valid,
                    "confidence": top_score,
                    "top_label": top_label,
                    "details": result,
                    "service_used": "huggingface"
                }
            
            raise Exception("Unexpected API response format")
            
        except Exception as e:
            print(f"Hugging Face API error: {str(e)}")
            return None, {"error": f"Classification failed: {str(e)}", "details": str(e)}
    
    def is_resume(self, text):
        """Try Hugging Face first, fall back to local method if it fails"""
        
        hf_result, hf_details = self.is_resume_hf(text)
        
        if hf_result is not None:  
            return hf_result, hf_details
        
        
        print("Falling back to local keyword analysis for classification")
        return self.is_resume_local(text)


class ValidateResumeView(APIView):
    permission_classes = [AllowAny]
    
    def post(self, request, format=None):
        file = request.FILES.get('file')
        if not file:
            return Response({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Make sure it's a PDF
        if not file.name.lower().endswith('.pdf'):
            return Response({"error": "File must be a PDF"}, status=status.HTTP_400_BAD_REQUEST)
        
        validator = ResumeValidator()
        
        try:
            file.seek(0)
            text = validator.extract_text(file)
            is_valid, results = validator.is_resume(text)
            
            return Response({
                'is_resume': is_valid,
                'confidence': results.get('confidence', 0),
                'top_label': results.get('top_label', ''),
                'service_used': results.get('service_used', 'unknown'),
                'details': results
            })
        except Exception as e:
            return Response(
                {"error": "Error validating document", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ChatMessagesView(APIView):
    # Allow any user to interact; if authenticated, we record their messages.
    permission_classes = [AllowAny]

    def get(self, request, format=None):
        """
        GET: Retrieve all chat messages for a given resume.
        Expects a query parameter 'resume_id'.
        """
        resume_id = request.query_params.get("resume_id")
        if not resume_id:
            return Response({"error": "Missing resume_id in query parameters."}, status=400)
        
        try:
            resume_obj = Resume.objects.get(id=resume_id)
        except Resume.DoesNotExist:
            return Response({"error": "Resume not found."}, status=404)
        
        messages = ChatMessage.objects.filter(resume=resume_obj).order_by("created_at")
        serializer = ChatMessageSerializer(messages, many=True)
        return Response(serializer.data)

    def post(self, request, format=None):
        """
        POST: Save a new chat message.
        Expected JSON payload: resume_id, message, sender ('user' or 'ai').
        """
        resume_id = request.data.get("resume_id")
        message = request.data.get("message")
        sender = request.data.get("sender")
        
        if not resume_id or not message or not sender:
            return Response({"error": "Missing resume_id, message, or sender."}, status=400)
        
        try:
            resume_obj = Resume.objects.get(id=resume_id)
        except Resume.DoesNotExist:
            return Response({"error": "Resume not found."}, status=404)
        
        chat_message = ChatMessage.objects.create(
            resume=resume_obj,
            user=request.user if request.user.is_authenticated else None,
            sender=sender,
            message=message
        )
        serializer = ChatMessageSerializer(chat_message)
        return Response(serializer.data, status=201)


    def post(self, request, format=None):
        """
        POST: Save a new chat message.
        Expects a JSON payload with:
          - resume_id: ID of the resume
          - message: The message text (for the user or AI)
          - sender: 'user' or 'ai'
        """
        resume_id = request.data.get("resume_id")
        message = request.data.get("message")
        sender = request.data.get("sender")
        
        if not resume_id or not message or not sender:
            return Response({"error": "Missing resume_id, message, or sender."}, status=status.HTTP_400_BAD_REQUEST)
        
        # Retrieve the resume
        try:
            resume_obj = Resume.objects.get(id=resume_id)
            # If no text has been extracted yet, try to extract from PDF
            if not resume_obj.text:
                with open(resume_obj.file.path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    extracted_text = ""
                    for page in pdf_reader.pages:
                        extracted_text += page.extract_text() or ""
                resume_obj.text = extracted_text
                resume_obj.save()
        except Resume.DoesNotExist:
            return Response({"error": "Resume not found."}, status=status.HTTP_404_NOT_FOUND)
        
        # Save the chat message. If the user is authenticated, record the user.
        chat_message = ChatMessage.objects.create(
            resume=resume_obj,
            user=request.user if request.user.is_authenticated else None,
            sender=sender,
            message=message
        )
        serializer = ChatMessageSerializer(chat_message)
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class UploadResumeView(APIView):
    permission_classes = [AllowAny]  # Allow any user to upload
    
    def post(self, request, format=None):
        file = request.FILES.get('file')
        validate_only = request.data.get('validate_only', False)
        
        if not file:
            return Response({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)
        
        file.seek(0)
        
        # Validate the resume first
        validator = ResumeValidator()
        try:
            file.seek(0)
            text = validator.extract_text(file)
            is_valid, results = validator.is_resume(text)
            
            # If validation only, return the results without saving
            if validate_only:
                return Response({
                    'is_resume': is_valid,
                    'confidence': results.get('confidence', 0),
                    'top_label': results.get('top_label', ''),
                    'details': results
                })
            
            # If not a resume, return error
            if not is_valid:
                return Response({
                    "error": "The uploaded file doesn't appear to be a resume",
                    "details": results
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Continue with upload if it's a valid resume
            resume = Resume(file=file)
            
            # If user is authenticated, associate the resume with them
            if request.user.is_authenticated:
                resume.user = request.user
            resume.save()
            
            # Extract and save text
            file.seek(0)
            pdf_reader = PyPDF2.PdfReader(file)
            try:
                extracted_text = ""
                for page in pdf_reader.pages:
                    extracted_text += page.extract_text() or ""
                resume.text = extracted_text
                resume.save()
            except Exception as e:
                return Response(
                    {"error": "Error processing PDF", "details": str(e)},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            
            serializer = ResumeSerializer(resume)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            return Response(
                {"error": "Error validating document", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
class ResumeDetailView(APIView):
    def get(self, request, pk):
        try:
            resume = Resume.objects.get(pk=pk)
            print(f"DEBUG: Resume with ID {pk} found.")
            serializer = ResumeSerializer(resume)
            print(f"DEBUG: data found. {serializer.data}")
            return Response(serializer.data)
            
        except Resume.DoesNotExist:
            return Response(
                {"error": "Resume not found"}, 
                status=status.HTTP_404_NOT_FOUND
            )

class AnalyzeResumeView(APIView):
    permission_classes = [AllowAny]

    def analyze_resume_ollama(self, resume_text):
        # Existing Ollama fallback code...
        pass

    def analyze_resume_github(self, resume_text):
        """Use GitHub-hosted model as fallback"""
        try:
            token = GITHUB_TOKEN
            endpoint = "https://models.inference.ai.azure.com"
            model_name = "mistral-small-2503"

            client = Mistral(api_key=token, server_url=endpoint)

            response = client.chat.complete(
                model=model_name,
                messages=[
                    SystemMessage(content="You are a helpful Resume ATS assistant."),
                    UserMessage(content=f"""
                                Analyze the following resume and return ONLY valid JSON (with no additional text or formatting) that exactly follows the structure below. Evaluate the resume and assign percentage scores (0–100) for each area
        scores (skills, experience, education, overall) as percentages, and. Also, provide exactly 10 key insights and exactly 10 actionable improvement suggestions referring to ATS. The key insights and improvement suggestions must cover the following areas:
        The expected JSON structure is:
        -  scores (skills, experience, education, overall),  key_insights (insight 1, insight 2, ... (exactly 10 insights) improvement_suggestions( suggestion 1,  suggestion 2,  ... (exactly 10 suggestions) )  ))
        - Formatting & Readability
        - Grammar & Language
        - Contact & Personal Information
        - Professional Summary or Objective
        - Skills & Competencies
        - Experience & Accomplishments
        - Education & Certifications
        - Keywords & ATS Optimization
        - Achievements & Awards
        - Projects & Publications (if applicable)
        - Overall Relevance & Customization
        - Consistency & Accuracy
        - Professional Tone & Branding
        - Red Flags & Gaps
        - Contact/Call-to-Action
        - Overall impression
        - Recommended jobs to consider based on this CV
        -
            Resume:{resume_text}
"""),
                ],
                temperature=1.0,
                max_tokens=1000,
                top_p=1.0
            )
            print(f"DEBUG: GitHub response: {response}")

            return response.choices[0].message.content
        except Exception as e:
            print(f"GitHub fallback failed: {str(e)}")
            return None

    def post(self, request, format=None):
        resume_id = request.data.get("resume_id")
        if not resume_id:
            return Response({"error": "No resume_id provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            resume = Resume.objects.get(id=resume_id)
        except Resume.DoesNotExist:
            return Response({"error": "Resume not found"}, status=status.HTTP_404_NOT_FOUND)

        if not resume.text:
            return Response({"error": "Resume text not found"}, status=status.HTTP_400_BAD_REQUEST)

        prompt = f"""
        Analyze the following resume and return ONLY valid JSON (with no additional text or formatting) that exactly follows the structure below. Evaluate the resume and assign percentage scores (0–100) for each area
        scores (skills, experience, education, overall) as percentages, and. Also, provide exactly 10 key insights and exactly 10 actionable improvement suggestions referring to ATS. The key insights and improvement suggestions must cover the following areas:
        The expected JSON structure is:
        -  scores (skills, experience, education, overall),  key_insights (insight 1, insight 2, ... (exactly 10 insights) improvement_suggestions( suggestion 1,  suggestion 2,  ... (exactly 10 suggestions) )  ))
        - Formatting & Readability
        - Grammar & Language
        - Contact & Personal Information
        - Professional Summary or Objective
        - Skills & Competencies
        - Experience & Accomplishments
        - Education & Certifications
        - Keywords & ATS Optimization
        - Achievements & Awards
        - Projects & Publications (if applicable)
        - Overall Relevance & Customization
        - Consistency & Accuracy
        - Professional Tone & Branding
        - Red Flags & Gaps
        - Contact/Call-to-Action
        - Overall impression
        - Recommended jobs to consider based on this CV
        -
            Resume: {resume.text}
              """

        headers = {
            "Authorization": f"Bearer {os.getenv('HF_API_KEY')}",
            "Content-Type": "application/json"
        }
        payload = {"inputs": prompt, "parameters": {"max_tokens": 10000}}

        try:
            hf_response = requests.post(
                "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
                headers=headers,
                json=payload
            )

            if hf_response.status_code == 200:
                result = hf_response.json()
                analysis = result[0].get("generated_text")
                json_start = analysis.find('{')
                json_end = analysis.rfind('}')
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    json_str = analysis[json_start:json_end+1]
                else:
                    json_str = analysis

                resume.analysis = json_str
                resume.save()

                serializer = ResumeSerializer(resume)
                return Response(serializer.data, status=status.HTTP_200_OK)
            else:
                print(f"Primary API failed with status {hf_response.status_code}. Trying fallbacks...")

                fallbacks = [self.analyze_resume_ollama, self.analyze_resume_github]

                for fallback_method in fallbacks:
                    try:
                        analysis = fallback_method(resume.text)
                        if analysis:
                            resume.analysis = analysis
                            resume.save()
                            serializer = ResumeSerializer(resume)
                            return Response(serializer.data, status=status.HTTP_200_OK)
                    except Exception as e:
                        print(f"Fallback method {fallback_method.__name__} failed: {str(e)}")
                        continue

                return Response(
                    {"error": "All analysis methods failed", "details": "Please try again later"},
                    status=status.HTTP_503_SERVICE_UNAVAILABLE
                )
        except Exception as e:
            return Response(
                {"error": "Error analyzing resume", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        


class ChatView(APIView):
    permission_classes = [AllowAny]

    def post(self, request, format=None):
        resume_id = request.data.get("resume_id")
        message = request.data.get("message")
        conversation = request.data.get("conversation", [])

        if not resume_id or not message:
            return Response({"error": "Missing resume_id or message"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            resume_obj = Resume.objects.get(id=resume_id)
            if resume_obj.text:
                resume_text = resume_obj.text
                print("DEBUG: Using stored resume text.")
            else:
                from PyPDF2 import PdfReader
                with open(resume_obj.file.path, 'rb') as f:
                    pdf_reader = PdfReader(f)
                    extracted_text = ""
                    for page in pdf_reader.pages:
                        extracted_text += page.extract_text() or ""
                resume_text = extracted_text
                resume_obj.text = extracted_text
                resume_obj.save()
                print("DEBUG: Extracted resume text from PDF.")
        except Resume.DoesNotExist:
            return Response({"error": "Resume not found"}, status=status.HTTP_404_NOT_FOUND)

        print("DEBUG: Resume text content:")
        print(resume_text)

        chat_prompt = (
            "You are an expert ATS resume advisor. Answer this person Your answer must reference specific details from the CV provided below. "
            "Do not provide generic advice. Instead, analyze the CV content (including skills, education, experience, achievements, etc.) "
            "and tailor your answer based on that information. If the CV lacks sufficient details, mention it explicitly. Do not exceed 100 words.\n\n"
            "CV Content:\n" + resume_text + "\n\n" +
            "Based on the CV above, please answer the person following the question, referencing specific details from the CV:\n" +
            "User: " + message + "\n" +
            "AI:"
        )
        token = GITHUB_TOKEN
        endpoint = "https://models.inference.ai.azure.com"
        model_name = "mistral-small-2503"

        client = Mistral(api_key=token, server_url=endpoint)

        try:
            ai_response = client.chat.complete(
                model=model_name,
                messages=[
                    SystemMessage(content="You are a helpful Resume ATS assistant."),
                    UserMessage(content=chat_prompt),
                ],
                temperature=1.0,
                max_tokens=1000,
                top_p=1.0
            )
            print(f"DEBUG: GitHub response: {ai_response}")

            # Access the correct attribute to get the generated text
            if ai_response.choices and len(ai_response.choices) > 0:
                reply = ai_response.choices[0].message.content
            else:
                reply = "Sorry, no response from AI."

            parts = reply.split("AI:")
            final_reply = parts[-1].strip() if parts else reply

            if request.user.is_authenticated:
                ChatMessage.objects.create(
                    resume=resume_obj,
                    user=request.user,
                    sender='ai',
                    message=final_reply
                )

            return Response({"reply": final_reply}, status=status.HTTP_200_OK)
        except Exception as e:
            print(f"DEBUG: Error processing AI response: {e}")
            return Response({"error": "Chat processing failed", "details": str(e)},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)



class SignupView(APIView):
    permission_classes = [AllowAny]
    
    def post(self, request, format=None):
        print("Signup attempt with data:", request.data)
        username = request.data.get("username")
        email = request.data.get("email")
        password = request.data.get("password")
        if not username or not email or not password:
            return Response(
                {"error": "Please provide username, email, and password."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if User.objects.filter(username=username).exists():
            return Response(
                {"error": "Username already exists."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            user = User.objects.create_user(username=username, email=email, password=password)
            token, created = Token.objects.get_or_create(user=user)
            return Response({
                "token": token.key,
                "user": {"username": user.username, "email": user.email}
            }, status=status.HTTP_201_CREATED)
        except Exception as e:
            print("Signup error:", str(e))
            return Response(
                {"error": "Signup failed", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class LoginView(APIView):
    permission_classes = [AllowAny]
    
    def post(self, request, format=None):
        print("Login attempt with data:", request.data)
        username = request.data.get("username")
        password = request.data.get("password")
        if not username or not password:
            return Response(
                {"error": "Please provide username and password."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        user = authenticate(username=username, password=password)
        if not user:
            return Response(
                {"error": "Invalid credentials."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        token, created = Token.objects.get_or_create(user=user)
        return Response({
            "token": token.key,
            "user": {"username": user.username, "email": user.email}
        }, status=status.HTTP_200_OK)
    
    
class LogoutView(APIView):
    permission_classes = [AllowAny]
    def post(self, request, format=None):
        token_key = request.data.get("token")
        if not token_key:
            return Response(
                {"error": "No token provided."},
                status=status.HTTP_400_BAD_REQUEST
            )
        try:
            token = Token.objects.get(key=token_key)
            token.delete()
            return Response({"message": "Successfully logged out."}, status=status.HTTP_200_OK)
        except Token.DoesNotExist:
            return Response({"error": "Invalid token."}, status=status.HTTP_400_BAD_REQUEST)



@api_view(['GET'])
@permission_classes([IsAuthenticated])
def account_detail(request):
    
    print("DEBUG: Request user:", request.user)
    print("DEBUG: Is authenticated?", request.user.is_authenticated)
    user = request.user
    data = {
        "name": user.username,
        "email": user.email,
        "phone": "",  # Populate if available
        "location": "",  # Populate if available
        "joined": user.date_joined.strftime("%B %d, %Y"),
        "avatar": "/api/placeholder/80/80"  # Replace with an actual avatar URL if available
    }
    return Response(data)



@api_view(['PUT'])
@permission_classes([IsAuthenticated])
def update_profile(request):
    
    user = request.user
    # Get new details from request data
    name = request.data.get("name")
    email = request.data.get("email")
    phone = request.data.get("phone")        # If your user model has extra fields, otherwise you'll need a custom profile model
    location = request.data.get("location")  # Same note as above

    # Update fields. For Django’s default User model, only username and email exist.
    if name:
        user.username = name
    if email:
        user.email = email
    user.save()

    # Return updated data. You can include phone and location if you have them.
    data = {
        "name": user.username,
        "email": user.email,
        "phone": phone if phone else "",        # Update as needed
        "location": location if location else "",
        "joined": user.date_joined.strftime("%B %d, %Y"),
        "avatar": "/api/placeholder/80/80"  # Replace with your actual avatar logic
    }
    return Response(data, status=status.HTTP_200_OK)

@api_view(['PUT'])
@permission_classes([IsAuthenticated])
def update_password(request):
    
    user = request.user
    current_password = request.data.get("currentPassword")
    new_password = request.data.get("newPassword")
    confirm_password = request.data.get("confirmPassword")
    
    if not current_password or not new_password or not confirm_password:
        return Response({"error": "All password fields are required."}, status=status.HTTP_400_BAD_REQUEST)
    
    if new_password != confirm_password:
        return Response({"error": "New password and confirm password do not match."}, status=status.HTTP_400_BAD_REQUEST)
    
    if not user.check_password(current_password):
        return Response({"error": "Current password is incorrect."}, status=status.HTTP_400_BAD_REQUEST)
    
    user.set_password(new_password)
    user.save()
    update_session_auth_hash(request, user)  # Keeps the user logged in after password change
    return Response({"message": "Password updated successfully."}, status=status.HTTP_200_OK)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_conversations(request):
    """
    Returns a summary list of conversations for the authenticated user.
    Each conversation is represented by a resume.
    """
    # Retrieve resumes uploaded by the user.
    resumes = Resume.objects.filter(user=request.user).order_by('-uploaded_at')
    conversations = [
        {
            "resume_id": resume.id,
            "resume_name": f"Resume {resume.id}"  # Replace with a proper title if available.
        }
        for resume in resumes
    ]
    return Response(conversations)



@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_account(request):
    user = request.user
    user.delete()  # This deletes the user record from the database.
    return Response({"message": "Account deleted successfully."}, status=200)



def clean_json_string(json_str):
    """
    Clean and prepare JSON string for parsing, handling common issues in AI-generated JSON.
    """
    # Replace escaped newlines with actual newlines
    cleaned = json_str.replace('\\n', '\n')
    # Handle escaped quotes properly
    cleaned = cleaned.replace('\\"', '"')
    # Replace double backslashes with single backslash
    cleaned = cleaned.replace('\\\\', '\\')

    # Remove invalid control characters
    cleaned = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', cleaned)

    # Find first { and last } to extract only one complete JSON object
    first_brace = cleaned.find('{')
    last_brace = cleaned.rfind('}')

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        # Extract what appears to be a single JSON object
        cleaned = cleaned[first_brace:last_brace+1]

    # Ensure property names are properly quoted
    pattern = r'{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:'
    cleaned = re.sub(pattern, r'{"\1":', cleaned)

    # This handles property names in the middle of JSON object
    pattern = r',\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:'
    cleaned = re.sub(pattern, r',"\1":', cleaned)

    return cleaned

@api_view(['POST'])
@permission_classes([AllowAny])
def rewrite_resume(request):
    """
    Endpoint to rewrite a resume using AI, with JSON output cleanup and escape sequence handling.
    """
    resume_id = request.data.get('resume_id')
    if not resume_id:
        return Response({'error': 'Resume ID not provided'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        # --- Get the original resume content ---
        resume = Resume.objects.get(id=resume_id)

        if not resume.text:
            try:
                if not resume.file or not hasattr(resume.file, 'path') or not os.path.exists(resume.file.path):
                    raise FileNotFoundError(
                        f"Resume file path not found or invalid: {getattr(resume.file, 'path', 'N/A')}"
                    )

                with open(resume.file.path, 'rb') as f:
                    pdf_reader = PdfReader(f)
                    if not pdf_reader.pages:
                        print(f"Warning: PDF for resume {resume_id} has no pages or is unreadable.")
                        raise ValueError("PDF contains no pages or is unreadable.")
                    extracted_text = ""
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            extracted_text += page_text + "\n"

                if not extracted_text.strip():
                    print(f"Warning: Could not extract text from PDF for resume {resume_id}.")
                    raise ValueError("Could not extract text from PDF.")

                resume.text = extracted_text
                resume.save(update_fields=['text'])
                print(f"DEBUG: Extracted text from PDF for resume {resume_id}")
            except (FileNotFoundError, ValueError, Exception) as extraction_error:
                print(f"Error extracting text from PDF for resume {resume_id}: {extraction_error}")
                error_detail = str(extraction_error)
                if isinstance(extraction_error, PyPDF2Errors.PdfReadError):
                    error_detail = "Could not read the PDF file; it might be corrupted or encrypted."
                return Response(
                    {'error': 'Failed to process the original resume PDF.', 'details': error_detail},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

        original_content = resume.text
        if not original_content or not original_content.strip():
            print(f"ERROR: Original resume content is empty for resume {resume_id}.")
            return Response({'error': 'Original resume content is empty.'}, status=status.HTTP_400_BAD_REQUEST)

        prompt = f"""
You are an expert ATS resume writer and formatter. Your task is to rewrite the provided raw resume text to be highly impactful, professional, ATS-optimized, and structured precisely in Markdown format.

Core Instructions:
1. Maintain Information: Preserve ALL original details (names, dates, companies, skills, descriptions, locations, contact details, etc.).
2. Enhance Wording: Improve clarity, use strong action verbs, quantify achievements, and ensure professional language.
3. ATS Optimization: Naturally integrate relevant keywords.
4. Markdown Structure: Format the rewritten resume using the specified Markdown structure below. Use '*' for ALL bullet points.
5. Output Format: Respond ONLY with a valid JSON object containing a single key "rewritten_markdown". The value MUST be a string containing the complete rewritten resume in Markdown, starting with '# Full Name'.
6. Strictness: Do NOT include any introductory text, explanations, or code block markers.

Markdown Template:
#[Full Name Extracted from Original]
[City, State (if available)] | [Phone Number (if available)] | [Email Address] | [LinkedIn Profile URL (if available, otherwise omit)]

## Summary
[Rewritten summary text...]

## Skills
*Programming Languages: [Comma-separated list]
*Frameworks & Libraries: [Comma-separated list]
*[...]

## Experience
###[Job Title]
**[Company Name] | [City, State] | [Start Month, Year] – [End Month, Year or Present]
*[Responsibility/achievement 1...]
*[Responsibility/achievement 2...]

###[Previous Job Title]
**[Previous Company Name] | [...]
*[...]

## Education
###[Degree Name]
**[Institution Name] | [...]
*[Optional bullet...]

## Projects (Include ONLY if distinct)
###[Project Name 1]
*[Description...]

## Certifications (Include ONLY if mentioned)
*[Certification Name...]

---

Original Resume Text (Raw):

{original_content}

Your Response (JSON Object Only):
"""

        # --- Send request to AI model ---
        token = GITHUB_TOKEN
        endpoint = "https://models.inference.ai.azure.com"
        model_name = "mistral-small-2503"

        client = Mistral(api_key=token, server_url=endpoint)
        rewritten_content = None

        try:
            print(f"DEBUG: Sending request to AI API for resume {resume_id}")
            response = client.chat.complete(
                model=model_name,
                messages=[
                    SystemMessage(content="You are an expert ATS resume writer and formatter."),
                    UserMessage(content=prompt),
                ],
                temperature=1.0,
                max_tokens=3000,
                top_p=1.0
            )
            print(f"DEBUG: AI API response: {response}")

            try:
                raw_generated_text = response.choices[0].message.content
                print(f"DEBUG: Raw AI response text for resume {resume_id}:\n---\n{raw_generated_text}\n---")
                cleaned_text = raw_generated_text.replace("```json", "").replace("```", "").strip()

                def extract_json_objects(text):
                    json_objects = []
                    json_pattern = r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
                    potential_jsons = re.findall(json_pattern, text)
                    for json_str in potential_jsons:
                        try:
                            parsed_json = json.loads(json_str)
                            if "rewritten_markdown" in parsed_json and isinstance(parsed_json["rewritten_markdown"], str):
                                json_objects.append(parsed_json)
                        except json.JSONDecodeError:
                            try:
                                fixed_json = json_str.replace("'", '"')
                                fixed_json = re.sub(r'(\{|\,)\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', fixed_json)
                                parsed_json = json.loads(fixed_json)
                                if "rewritten_markdown" in parsed_json and isinstance(parsed_json["rewritten_markdown"], str):
                                    json_objects.append(parsed_json)
                                    print(f"DEBUG: Fixed and parsed JSON successfully.")
                            except (json.JSONDecodeError, Exception) as fix_error:
                                print(f"DEBUG: Failed to fix JSON: {fix_error}")
                                continue
                    return json_objects

                json_objects = extract_json_objects(cleaned_text)
                if json_objects:
                    rewritten_content = json_objects[0]["rewritten_markdown"].strip()
                    print(f"DEBUG: Successfully extracted JSON with {len(json_objects)} valid objects.")
                else:
                    print("DEBUG: No valid JSON objects found; attempting to extract markdown directly.")
                    markdown_markers = ["# ", "## ", "#"]
                    content_lines = cleaned_text.splitlines()
                    for i, line in enumerate(content_lines):
                        if any(line.strip().startswith(marker) for marker in markdown_markers):
                            potential_content = "\n".join(content_lines[i:])
                            if potential_content.strip():
                                rewritten_content = potential_content
                                print(f"DEBUG: Extracted markdown content starting with: {line.strip()}")
                                break
                    if not rewritten_content and "{" in cleaned_text and "}" in cleaned_text:
                        json_start = cleaned_text.find('{')
                        json_end = cleaned_text.rfind('}')
                        if json_start != -1 and json_end != -1 and json_end > json_start:
                            extracted_text = cleaned_text[json_start:json_end + 1]
                            markdown_pattern = r'"rewritten_markdown"\s*:\s*"(.*?)"(?=\s*[,}])'
                            markdown_matches = re.findall(markdown_pattern, extracted_text, re.DOTALL)
                            if markdown_matches:
                                potential_content = markdown_matches[0]
                                potential_content = potential_content.replace('\\"', '"').replace('\\n', '\n')
                                if potential_content.strip():
                                    rewritten_content = potential_content
                                    print("DEBUG: Extracted markdown using regex pattern.")
                if not rewritten_content and cleaned_text.startswith('#') and '\n##' in cleaned_text:
                    rewritten_content = cleaned_text
                    print("DEBUG: Using raw text as markdown directly as last resort.")

            except Exception as parse_error:
                print(f"Warning: Failed to parse AI response content: {parse_error}")
                traceback.print_exc()
        except requests.exceptions.RequestException as api_error:
            print(f"Error calling AI API: {api_error}")
            if hasattr(api_error, 'response') and api_error.response is not None:
                print(f"API Error Status: {api_error.response.status_code}")
                print(f"API Error Body: {api_error.response.text[:500]}...")
        except Exception as e:
            print(f"Unexpected error during AI processing: {e}")
            traceback.print_exc()

        if not rewritten_content or not rewritten_content.strip():
            print("ERROR: Failed to process AI response. Using mock data.")
            rewritten_content = (
                "# JOHN DOE\n"
                "New York, NY | (555) 123-4567 | johndoe@email.com | linkedin.com/in/johndoe\n\n"
                "## Summary\n"
                "Results-driven software engineer with 5+ years of experience building scalable web applications. "
                "Expertise in React, Node.js, and cloud architecture. Strong problem-solving skills with a focus on delivering high-quality code and excellent user experiences.\n\n"
                "## Skills\n"
                "* **Programming Languages:** JavaScript, Python, TypeScript, SQL\n"
                "* **Frameworks & Libraries:** React, Node.js, Express, Django, Redux, TailwindCSS\n"
                "* **Tools & Platforms:** Git, Docker, AWS, CI/CD, Jira, Agile methodologies\n\n"
                "## Experience\n"
                "### SENIOR SOFTWARE ENGINEER\n"
                "**ABC Tech** | New York, NY | January 2020 – Present\n"
                "* Led development of new customer portal that improved user engagement by 35%.\n"
                "* Architected microservice infrastructure that reduced deployment time by 40%.\n"
                "* Mentored 5 junior developers through code reviews and technical training.\n\n"
                "### SOFTWARE ENGINEER\n"
                "**XYZ Solutions** | Boston, MA | June 2017 – December 2019\n"
                "* Developed RESTful APIs that increased revenue by 20%.\n"
                "* Optimized database queries to reduce page load times by 60%.\n"
                "* Implemented an automated testing suite increasing code coverage from 65% to 92%.\n\n"
                "## Education\n"
                "### MASTER OF SCIENCE IN COMPUTER SCIENCE\n"
                "**Massachusetts Institute of Technology** | Cambridge, MA | 2017\n\n"
                "### BACHELOR OF SCIENCE IN COMPUTER ENGINEERING\n"
                "**University of California, Berkeley** | Berkeley, CA | 2015\n\n"
                "[Note: This is mock data as the AI service response could not be processed reliably.]"
            )
            message = "Resume rewrite failed to process AI response, using placeholder data."
        else:
            message = "Resume rewrite processed successfully using AI response."

        try:
            resume.rewritten_content = rewritten_content
            resume.save(update_fields=['rewritten_content'])
            print(f"DEBUG: Saved rewritten content (length: {len(rewritten_content)}) for resume {resume_id}")
        except Exception as db_error:
            print(f"ERROR: Failed to save rewritten content for resume {resume_id}: {db_error}")
            return Response(
                {'error': 'Failed to save the rewritten resume content.', 'details': str(db_error)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        return Response({
            'rewritten_content': rewritten_content,
            'message': message
        }, status=status.HTTP_200_OK)

    except Resume.DoesNotExist:
        return Response({'error': 'Resume not found'}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        print(f"Fatal error in rewrite_resume view for resume {resume_id}: {e}")
        traceback.print_exc()
        return Response(
            {'error': 'An unexpected server error occurred.', 'details': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )






@api_view(['POST'])
@permission_classes([AllowAny])
def revise_resume(request):
    """
    Endpoint to revise a rewritten resume based on user feedback,
    with improved JSON handling and error recovery identical to rewrite_resume.
    """
    resume_id = request.data.get('resume_id')
    feedback = request.data.get('feedback')
    current_version = request.data.get('current_version')
    
    if not resume_id:
        return Response({'error': 'Resume ID not provided'}, status=status.HTTP_400_BAD_REQUEST)
    
    if not feedback:
        return Response({'error': 'Feedback not provided'}, status=status.HTTP_400_BAD_REQUEST)
    
    if not current_version:
        return Response({'error': 'Current resume version not provided'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        # Get the resume
        resume = Resume.objects.get(id=resume_id)
        
        # Create prompt for Mistral client
        prompt = f"""
You are an expert ATS resume writer and formatter. Your task is to revise the provided resume based on the user's feedback, while maintaining professional ATS formatting and style.

Core Instructions:
1.  Make Requested Changes: Apply the user's feedback carefully, preserving the overall professional quality.
2.  Maintain Information: Preserve ALL original information that the user doesn't ask to change.
3.  Enhance Wording: Improve clarity, use strong action verbs, quantify achievements, and ensure professional language.
4.  ATS Optimization: Naturally integrate relevant keywords.
5.  Markdown Structure: Format the revised resume using the standard Markdown structure provided below.
6.  Output Format: Respond ONLY with a valid JSON object containing a single key "revised_markdown". The value associated with this key MUST be a string containing the complete, revised resume in Markdown format, starting directly with the '# Full Name' heading.
7.  Strictness: Do NOT include any introductory text, explanations, apologies, code block markers (like ```json), or any text whatsoever before or after the single JSON object in your response.

Markdown Structure Template (for the value of "revised_markdown"):

#[Full Name]
[City, State (if available)] | [Phone Number (if available)] | [Email Address] | [LinkedIn Profile URL (if available, otherwise omit)]

##Summary
[Revised summary text...]

##Skills
*Programming Languages: [Comma-separated list]
*Frameworks & Libraries: [Comma-separated list]
*[...]

##Experience
###[Job Title]
**[Company Name] | [City, State] | [Start Month, Year] – [End Month, Year or Present]
*[Revised responsibility/achievement 1...]
*[Revised responsibility/achievement 2...]

###[Previous Job Title]
**[Previous Company Name] | [...]
*[...]

##Education
###[Degree Name]
**[Institution Name] | [...]
*[Optional bullet...]

##Projects (Include ONLY if distinct)
###[Project Name 1]
*[Description...]

##Certifications (Include ONLY if mentioned)
*[Certification Name...]

---

Current Resume:
{current_version}

User Feedback:
{feedback}

Your Response (JSON Object Only):
"""
        
        # Setup Mistral client
        token = GITHUB_TOKEN
        endpoint = "https://models.inference.ai.azure.com"
        model_name = "mistral-small-2503"

        client = Mistral(api_key=token, server_url=endpoint)
        
        revised_content = None  
        
        try:
            print(f"DEBUG: Sending request to Mistral API for resume revision {resume_id}")
            response = client.chat.complete(
                model=model_name,
                messages=[
                    SystemMessage(content="You are an expert ATS resume writer and formatter."),
                    UserMessage(content=prompt),
                ],
                temperature=0.7,
                max_tokens=3000,
                top_p=1.0
            )
            print(f"DEBUG: Mistral API response: {response}")
            
            try:
                # Get the raw text directly from the Mistral response
                raw_generated_text = response.choices[0].message.content
                print(f"DEBUG: Raw AI Response Text for resume revision {resume_id}:\n---\n{raw_generated_text}\n---")
                
                # Enhanced JSON handling
                # First, clean the raw text - remove any potential code block markers
                cleaned_text = raw_generated_text.replace("```json", "").replace("```", "").strip()
                
                # Function to extract and validate JSON objects from text
                def extract_json_objects(text):
                    json_objects = []
                    # Find all potential JSON objects in the text
                    json_pattern = r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
                    potential_jsons = re.findall(json_pattern, text)
                    
                    for json_str in potential_jsons:
                        try:
                            parsed_json = json.loads(json_str)
                            if "revised_markdown" in parsed_json and isinstance(parsed_json["revised_markdown"], str):
                                json_objects.append(parsed_json)
                        except json.JSONDecodeError:
                            # Try to fix common JSON formatting errors
                            try:
                                # Handle cases where single quotes are used instead of double quotes
                                fixed_json = json_str.replace("'", '"')
                                # Handle unquoted keys
                                fixed_json = re.sub(r'(\{|\,)\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', fixed_json)
                                parsed_json = json.loads(fixed_json)
                                if "revised_markdown" in parsed_json and isinstance(parsed_json["revised_markdown"], str):
                                    json_objects.append(parsed_json)
                                    print(f"DEBUG: Fixed and parsed JSON successfully: {fixed_json[:100]}...")
                            except (json.JSONDecodeError, Exception) as fix_error:
                                print(f"DEBUG: Failed to fix JSON: {fix_error}")
                                continue
                    
                    return json_objects
                
                # Try to extract valid JSON objects
                json_objects = extract_json_objects(cleaned_text)
                
                if json_objects:
                    # Use the first valid JSON object found
                    revised_content = json_objects[0]["revised_markdown"].strip()
                    print(f"DEBUG: Successfully extracted JSON. Found {len(json_objects)} valid JSON objects.")
                else:
                    # If no valid JSON found, try extracting markdown directly
                    print("DEBUG: No valid JSON objects found. Attempting to extract markdown directly.")
                    
                    # Look for common markdown headers that would indicate the start of a resume
                    markdown_starts = ["# ", "## ", "#"]
                    content_lines = cleaned_text.splitlines()
                    
                    for i, line in enumerate(content_lines):
                        line = line.strip()
                        if any(line.startswith(marker) for marker in markdown_starts):
                            # Found what appears to be the start of markdown content
                            potential_content = "\n".join(content_lines[i:])
                            if potential_content.strip():
                                revised_content = potential_content
                                print(f"DEBUG: Extracted markdown content starting with: {line}")
                                break
                    
                    if not revised_content and "{" in cleaned_text and "}" in cleaned_text:
                        # Try one more approach - extract all text between first { and last }
                        json_start = cleaned_text.find('{')
                        json_end = cleaned_text.rfind('}')
                        
                        if json_start != -1 and json_end != -1 and json_end > json_start:
                            extracted_text = cleaned_text[json_start:json_end+1]
                            
                            # Use a simple string-based approach to extract the markdown content
                            markdown_pattern = r'"revised_markdown"\s*:\s*"(.*?)"(?=\s*[,}])'
                            markdown_matches = re.findall(markdown_pattern, extracted_text, re.DOTALL)
                            
                            if markdown_matches:
                                # Use the first match
                                potential_content = markdown_matches[0]
                                # Unescape escaped quotes and newlines
                                potential_content = potential_content.replace('\\"', '"').replace('\\n', '\n')
                                if potential_content.strip():
                                    revised_content = potential_content
                                    print("DEBUG: Extracted markdown using regex pattern.")
                
                if not revised_content:
                    # One final attempt: if the response looks like markdown directly
                    if cleaned_text.startswith('#') and '\n##' in cleaned_text:
                        revised_content = cleaned_text
                        print("DEBUG: Using raw text as markdown directly as last resort.")
                        
            except Exception as parse_error:
                print(f"Warning: Failed to parse content from AI response ({parse_error}). Attempting fallback.")
                import traceback
                traceback.print_exc()
                
        except requests.exceptions.RequestException as api_error:
            print(f"Error calling Mistral API: {api_error}")
            
            if hasattr(api_error, 'response') and api_error.response is not None:
                print(f"API Error Response Status: {api_error.response.status_code}")
                print(f"API Error Response Body: {api_error.response.text[:500]}...")  # Log first 500 chars
            
        except Exception as e:
            print(f"Unexpected error during AI processing: {e}")
            import traceback
            traceback.print_exc()  
           
        if revised_content is None or not revised_content.strip():  # Check if it's None or empty/whitespace
            print("ERROR: Failed to get valid content from AI after JSON and fallback attempts. Using existing content with feedback note.")
            
            revised_content = current_version + f"\n\n# REVISION BASED ON FEEDBACK: \n{feedback}\n\n[Note: This is a placeholder as the AI service response could not be processed reliably.]"
            message = "Resume revision failed to process AI response, using placeholder data with feedback note."
        else:
            message = "Resume revision processed successfully using AI response."
            
        # Clean up excessive newlines and bullet points
        revised_content = re.sub(r'\n{3,}', '\n\n', revised_content)  
        revised_content = re.sub(r'(?<=\n)\s*\*\s+', '* ', revised_content)
        
        try:
            resume.rewritten_content = revised_content
            resume.save(update_fields=['rewritten_content'])
            print(f"DEBUG: Saved revised content (length: {len(revised_content)}) for resume {resume_id}")
        except Exception as db_error:
            print(f"ERROR: Failed to save revised content to database for resume {resume_id}: {db_error}")
            return Response(
                {'error': 'Failed to save the revised resume content.', 'details': str(db_error)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
            
        return Response({
            'revised_content': revised_content,
            'message': message  
        }, status=status.HTTP_200_OK)
        
    except Resume.DoesNotExist:
        return Response(
            {'error': 'Resume not found'},
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        print(f"Fatal error in revise_resume view for resume {resume_id}: {e}")
        import traceback
        traceback.print_exc()  
        return Response(
            {'error': 'An unexpected server error occurred.', 'details': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

try:
    from markdown import markdown
except ImportError:
    markdown = None
    print("ERROR: 'markdown' library not found. pip install markdown")

try:
    from xhtml2pdf import pisa
except ImportError:
    pisa = None
    print("ERROR: 'xhtml2pdf' library not found. pip install xhtml2pdf")


@api_view(['POST'])
@permission_classes([AllowAny])
def generate_pdf(request):
    """
    Endpoint to generate a PDF from markdown content using xhtml2pdf,
    with improved preprocessing of markdown content for better rendering.
    """
    if not markdown or not pisa:
        return Response(
            {'error': 'Required PDF generation libraries (markdown, xhtml2pdf) are missing on the server.'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    content_md = request.data.get('content', '')
    if not content_md:
        return Response({'error': 'No content provided'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        # 0. Preprocess markdown for better conversion
        print("DEBUG: Starting markdown preprocessing")
        
        content_md = content_md.replace("```markdown", "").replace("```json", "").replace("```", "")
        
        content_md = re.sub(r'\n{3,}', '\n\n', content_md)  # Replace excessive newlines
        
        content_md = re.sub(r'(?<=\n)\s*\*\s+', '* ', content_md)
        
        content_md = re.sub(r'(?<=\n)#{1,6}\s*', lambda m: m.group().strip() + ' ', content_md)
        
        print("DEBUG: Starting Markdown to HTML conversion.")
        html_content = markdown(content_md, extensions=['fenced_code', 'tables', 'nl2br'])
        print("DEBUG: Markdown to HTML conversion finished.")

        print("DEBUG: Adding structure for PDF formatting")
       
        html_content = re.sub(r'<h3>(.*?)</h3>', r'<h3 class="job-title">\1</h3>', html_content)
        
        sections = re.findall(r'<h3 class="job-title">.*?(?=<h3 class="job-title">|<h2|$)', html_content, re.DOTALL)
        for section in sections:
            html_content = html_content.replace(section, f'<div class="job-entry">{section}</div>')
        
        html_content = re.sub(r'<h3>(.*?University.*?|.*?College.*?|.*?School.*?|.*?Institute.*?)</h3>', 
                             r'<h3 class="education-title">\1</h3>', html_content)
        
        sections = re.findall(r'<h3 class="education-title">.*?(?=<h3|<h2|$)', html_content, re.DOTALL)
        for section in sections:
            html_content = html_content.replace(section, f'<div class="education-entry">{section}</div>')
        
        css_content = """
        @page {
            size: letter;
            margin: 0.75in;
        }

        html {
            font-variant-ligatures: common-ligatures;
        }

        body {
            font-family: "Helvetica", "Arial", sans-serif;
            font-size: 10pt;
            line-height: 1.4;
            color: #333;
            /* Ensure content is allowed to flow without fixed constraints */
            overflow: visible;
        }

        h1, h2, h3, h4, h5, h6 {
            font-weight: bold;
            color: #000;
            margin-top: 1.2em;
            margin-bottom: 0.6em;
            /* Avoid forcing elements into the same page if possible */
            page-break-after: avoid;
            page-break-before: avoid;
        }

        h1 {
            font-size: 18pt;
            margin-top: 0;
            text-align: center;
        }

        h2 {
            font-size: 14pt;
            border-bottom: 1px solid #eee;
            padding-bottom: 3pt;
        }

        h3 {
            font-size: 11pt;
        }

        p {
            margin-top: 0;
            margin-bottom: 0.8em;
            text-align: left;
            orphans: 3;
            widows: 3;
            /* Let paragraphs break naturally */
            page-break-inside: avoid;
        }

        ul, ol {
            padding-left: 20pt;
            margin-top: 0.5em;
            margin-bottom: 0.8em;
        }

        li {
            margin-bottom: 0.4em;
        }

        .job-entry, .education-entry, .project-entry {
            page-break-inside: avoid;
            margin-bottom: 1.5em;
        }

        p.contact-info {
            text-align: center;
            font-size: 9pt;
            margin-bottom: 1.5em;
            color: #555;
            border-bottom: 1px solid #ccc;
            padding-bottom: 8pt;
            page-break-after: avoid;
        }

        p.location-date {
            font-size: 10pt;
            font-style: italic;
            color: #666;
            margin-top: -0.4em;
            margin-bottom: 0.6em;
        }

        strong, b {
            font-weight: bold;
        }

        em, i {
            font-style: italic;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 1em;
        }

        th, td {
            border: 1px solid #ddd;
            padding: 6pt;
            text-align: left;
        }

        thead {
            display: table-header-group;
            font-weight: bold;
            background-color: #f2f2f2;
        }

        tr {
            page-break-inside: avoid;
            page-break-after: auto;
        }
        """
        
        # Combine HTML and CSS
        full_html = f"<!DOCTYPE html><html><head><meta charset=\"UTF-8\"><style>{css_content}</style></head><body>{html_content}</body></html>"

        # --- SAVE INTERMEDIATE HTML FOR DEBUGGING ---
        try:
            
            debug_dir = os.path.join(settings.BASE_DIR, 'debug_output') # Or settings.MEDIA_ROOT
            os.makedirs(debug_dir, exist_ok=True)
            debug_html_path = os.path.join(debug_dir, "debug_resume_output.html")
            with open(debug_html_path, "w", encoding="utf-8") as f:
                f.write(full_html)
            print(f"DEBUG: Saved intermediate HTML to {debug_html_path}")
        except Exception as html_save_error:
            print(f"Warning: Could not save debug HTML file: {html_save_error}")
        
        print("DEBUG: Starting PDF generation with pisa.")
        pdf_buffer = io.BytesIO()
        pisa_status = pisa.CreatePDF(
            src=io.BytesIO(full_html.encode('UTF-8')),
            dest=pdf_buffer,
            encoding='UTF-8'
            
        )
        print(f"DEBUG: pisa.CreatePDF finished. Success: {not pisa_status.err}")

        if pisa_status.err:
            print(f"xhtml2pdf Error Code: {pisa_status.err}") 
            error_details = f"PDF Generation Error Code: {pisa_status.err}"
            detailed_log = pisa_status.log 
            if detailed_log:
                 print("--- xhtml2pdf Log Messages ---")
                 log_str = ""
                 for msg_type, msg, line, col in detailed_log:
                      log_line = f"Type: {msg_type}, Line: {line}, Col: {col}, Msg: {msg}"
                      print(log_line)
                      log_str += log_line + "\n"
                 print("-----------------------------")
                 error_details += "\nLog:\n" + log_str
            else:
                 print("No detailed log messages available from xhtml2pdf.")

           
            error_details += f"\n\n--- Failing HTML (approx first 500 chars) ---\n{full_html[:500]}"

            return Response({'error': 'PDF generation failed', 'details': error_details}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        print("DEBUG: PDF generated successfully. Preparing response.")
        pdf_buffer.seek(0)
        response = HttpResponse(pdf_buffer, content_type='application/pdf')
        response['Content-Disposition'] = 'attachment; filename="aires_revised.pdf"'
        return response

    except Exception as e:
        
        print(f"Error in generate_pdf view: {e}")
        import traceback
        traceback.print_exc()
        return Response({'error': 'An internal server error occurred during PDF generation.', 'details': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)