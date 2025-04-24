import os, io
from rest_framework.views import APIView
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework import status
from .models import (
    Resume, Analysis, Conversation, Message, UpgradedResume,
    JobRequirement, ChatMessage, RecruiterAnalysis, CandidateResume, USER_ROLE_CHOICES, MESSAGE_SENDER_CHOICES   )
from .serializers import ResumeSerializer, ChatMessageSerializer
import PyPDF2
import requests
import json
from django.http import JsonResponse
from django.conf import settings
from django.contrib.auth.models import User, Group
from django.contrib.auth import authenticate
from rest_framework.authtoken.models import Token
from django.contrib.auth import update_session_auth_hash
from django.db import IntegrityError, transaction
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


GITHUB_TOKEN="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"


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

# views.py

# --- Essential Django/DRF Imports ---
import os
import io
import json
import re
import traceback
from django.conf import settings
from django.db import transaction, IntegrityError # For atomic DB operations
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated # Enforce Login

# --- External Libraries ---
import requests
try:
    # For PDF Reading
    from PyPDF2 import PdfReader, errors as PyPDF2Errors
except ImportError:
    PdfReader = None
    PyPDF2Errors = None
    print("CRITICAL WARNING: PyPDF2 library not found. PDF processing will be unavailable.")

try:
    # For AI Interaction
    from mistralai import Mistral, UserMessage, SystemMessage
except ImportError:
    Mistral = None
    UserMessage = None
    SystemMessage = None
    print("WARNING: mistralai library not found. AI analysis will fail.")

# --- Project Model Imports ---
from .models import RecruiterAnalysis, CandidateResume # Import required models
# Potentially JobRequirement if you want to link/create it here, but not strictly needed for saving analysis

# --- Constants ---
# Load API key securely from environment variable is best practice
GITHUB_TOKEN = os.getenv("MISTRAL_API_KEY", "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") # Replace fallback in production
MAX_RECRUITER_FILES = 5 # Should match frontend setting

# --- Full RecruiterAnalyzeView Class ---

class RecruiterAnalyzeView(APIView):
    """
    Handles the analysis of multiple candidate resumes against a job description
    for an authenticated recruiter. Extracts text, calls AI for comparison,
    parses the results, saves the analysis and associated resumes to the database,
    and returns the analysis results to the frontend.
    """
    permission_classes = [IsAuthenticated] # Require user to be logged in

    def extract_text_from_pdf(self, pdf_file_obj):
        """Extracts text from an InMemoryUploadedFile or similar file object."""
        if PdfReader is None:
             raise ImportError("PyPDF2 library is required for PDF processing.")

        text = ""
        try:
            pdf_file_obj.seek(0) # Ensure reading from the start
            pdf_reader = PdfReader(pdf_file_obj)
            if not pdf_reader.pages:
                 print(f"Warning: PDF file {getattr(pdf_file_obj, 'name', 'N/A')} has no pages or is unreadable.")
                 return None # Indicate failure

            for page in pdf_reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as page_extract_error:
                    # Log warning but continue if possible
                    print(f"Warning: Could not extract text from one page in {getattr(pdf_file_obj, 'name', 'N/A')}: {page_extract_error}")
                    continue # Try next page

            pdf_file_obj.seek(0) # Reset seek for potential later use (like saving)

            if not text.strip():
                 print(f"Warning: Could not extract any text from PDF {getattr(pdf_file_obj, 'name', 'N/A')}.")
                 return None # Indicate failure

            return text
        except PyPDF2Errors.PdfReadError as pdf_err:
            file_name = getattr(pdf_file_obj, 'name', 'N/A')
            print(f"Error reading PDF {file_name}: {pdf_err}")
            if "encrypted" in str(pdf_err).lower():
                raise ValueError(f"File '{file_name}' is encrypted and cannot be processed.")
            else:
                raise ValueError(f"File '{file_name}' is corrupted or not a valid PDF.")
        except Exception as e:
            file_name = getattr(pdf_file_obj, 'name', 'N/A')
            print(f"Unexpected error extracting text from {file_name}: {e}")
            traceback.print_exc()
            raise ValueError(f"Failed to process file '{file_name}'.")

    def clean_ai_json_response(self, raw_text):
        """Attempts to extract a valid JSON object from the AI's raw text output."""
        if not isinstance(raw_text, str):
            return None # Input must be a string

        # Remove markdown code fences
        cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw_text, flags=re.MULTILINE | re.DOTALL).strip()

        # Basic check for JSON structure
        if not cleaned.startswith('{') or not cleaned.endswith('}'):
            print("WARN: Cleaned text doesn't start/end with braces. JSON parsing likely to fail.")
            # Attempt to find the first '{' and last '}' anyway as a fallback
            first_brace = cleaned.find('{')
            last_brace = cleaned.rfind('}')
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                 cleaned = cleaned[first_brace : last_brace + 1]
            else:
                 print("ERROR: Cannot find JSON object structure.")
                 return None # Cannot find even basic structure

        try:
            # Standard JSON parsing
            # Handle potential escape issues that might remain
            cleaned_corrected = cleaned.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
            # Remove control characters which can break parsing
            cleaned_corrected = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', cleaned_corrected)

            parsed_json = json.loads(cleaned_corrected)
            return parsed_json
        except json.JSONDecodeError as e:
            print(f"ERROR: JSON Decode Error after cleaning: {e}")
            print(f"Problematic JSON string segment (first 500 chars):\n {cleaned[:500]}...")
            # Add more advanced cleaning/fixing here if needed (e.g., trailing commas)
            return None # Failed to parse

    # Use atomic transaction for database operations
    @transaction.atomic
    def post(self, request, format=None):
        """
        Handles POST request to analyze resumes. Requires authentication.
        Saves results to RecruiterAnalysis and CandidateResume models.
        """
        if Mistral is None:
             return Response({'error': 'AI analysis service is not configured correctly on the server.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        job_description = request.data.get('job_description')
        resume_files = request.FILES.getlist('resumes') # List of InMemoryUploadedFile

        # --- Input Validation ---
        if not job_description or not job_description.strip():
            return Response({'error': 'Job description is required.'}, status=status.HTTP_400_BAD_REQUEST)
        if not resume_files:
            return Response({'error': 'At least one resume file (PDF) is required.'}, status=status.HTTP_400_BAD_REQUEST)
        if len(resume_files) > MAX_RECRUITER_FILES:
            return Response({'error': f'A maximum of {MAX_RECRUITER_FILES} resume files can be uploaded.'}, status=status.HTTP_400_BAD_REQUEST)

        # --- Process Resumes: Extract Text and Store File Objects ---
        processed_resumes_data = [] # List to store tuples: (file_obj, identifier, extracted_text)
        processing_errors = [] # Collect non-fatal errors for optional reporting

        print(f"DEBUG: Processing {len(resume_files)} uploaded files...")
        for index, file_obj in enumerate(resume_files):
            file_identifier = file_obj.name # Use filename as the unique ID for this batch

            # Check file type
            if not file_identifier.lower().endswith('.pdf'):
                 warning_msg = f"File '{file_identifier}' is not a PDF and was skipped."
                 print(f"WARN: {warning_msg}")
                 processing_errors.append(warning_msg)
                 continue # Skip non-PDF files

            # Attempt text extraction
            try:
                extracted_text = self.extract_text_from_pdf(file_obj)
                if extracted_text is None:
                    warning_msg = f"Could not extract text from '{file_identifier}' (might be image-based or corrupted); it was skipped."
                    print(f"WARN: {warning_msg}")
                    processing_errors.append(warning_msg)
                    continue # Skip files where text extraction failed

                # Store successful data: the file object, its identifier, and the text
                processed_resumes_data.append((file_obj, file_identifier, extracted_text))
                print(f"DEBUG: Successfully processed '{file_identifier}'. Text length: {len(extracted_text)}")

            except ValueError as extraction_error:
                # Critical errors raised from extract_text_from_pdf (e.g., encrypted)
                print(f"ERROR: Value error processing '{file_identifier}': {extraction_error}")
                return Response({'error': str(extraction_error)}, status=status.HTTP_400_BAD_REQUEST)
            except ImportError as import_err:
                 # Raised if PyPDF2 wasn't loaded
                 print(f"ERROR: PDF processing library unavailable: {import_err}")
                 return Response({'error': 'PDF processing library is unavailable on the server.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            except Exception as e:
                 # Catch unexpected errors during file processing
                 print(f"ERROR: Unexpected error processing file '{file_identifier}': {e}")
                 traceback.print_exc()
                 return Response({'error': f"An unexpected server error occurred while processing '{file_identifier}'."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Check if any resumes were successfully processed
        if not processed_resumes_data:
             error_detail = " ".join(processing_errors) if processing_errors else "No valid PDF resumes were found or could be processed."
             return Response({'error': 'No resumes could be processed.', 'details': error_detail}, status=status.HTTP_400_BAD_REQUEST)

        print(f"DEBUG: Proceeding to AI analysis with {len(processed_resumes_data)} processed resumes.")

        # --- Construct AI Prompt ---
        resume_sections = ""
        file_identifiers_for_prompt = []
        for i, (_, identifier, text) in enumerate(processed_resumes_data):
            resume_sections += f"--- Resume {i+1} (Identifier: {identifier}) ---\n"
            resume_sections += f"{text}\n\n"
            file_identifiers_for_prompt.append(identifier)

        # Use the detailed, structured JSON prompt (ensure it matches the expected models/fields)
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
1.  **Overall Evaluation:** For EACH resume, provide an overall `match_score` (integer 0-100)...
2.  **Categorical Scores:** For EACH resume, provide specific scores (integer 0-100) for: `skills_score`, `experience_score`, `education_score`, `keyword_score`.
3.  **Keyword Analysis:** For EACH resume: list top `keywords_matched` and `keywords_missing`.
4.  **Strengths & Weaknesses:** For EACH resume, provide concise lists (3-5 points each) of `strengths` and `weaknesses` relative to the job.
5.  **Red Flags:** For EACH resume, list potential `red_flags` (0-3 points). If none, provide an empty list [].
6.  **Recommendation Tier:** For EACH resume, assign `recommendation_tier` ("Top Match", "Strong Candidate", "Potential Fit", "Less Suitable").
7.  **AI Summaries:** Provide `job_description_summary` (1-2 sentences) and a concise `summary` (2-3 sentences) for EACH candidate vs the role.
8.  **Comparative Analysis:** Provide a `comparative_analysis` section (1-2 paragraphs) comparing candidates head-to-head for this role.
9.  **Ranking:** Provide a `ranking` list containing the exact `resume_identifier` strings (provided in the input) ordered from most suitable to least suitable.
10. **Output Format:** Respond ONLY with a single, valid JSON object adhering strictly to the structure below. Ensure all strings are properly escaped. No text outside the JSON.

Expected JSON Structure:
{{
  "job_description_summary": "string",
  "comparative_analysis": "string",
  "ranking": {json.dumps(file_identifiers_for_prompt)}, // Example embedding identifiers
  "candidate_analysis": [
    {{
      "resume_identifier": "{file_identifiers_for_prompt[0] if file_identifiers_for_prompt else 'example_identifier_1'}", // Use actual identifier
      "match_score": integer,
      "recommendation_tier": "string",
      "summary": "string",
      "details": {{
        "skills_score": integer, "experience_score": integer, "education_score": integer, "keyword_score": integer,
        "keywords_matched": ["string", ...], "keywords_missing": ["string", ...],
        "strengths": ["string", ...], "weaknesses": ["string", ...],
        "red_flags": ["string", ...] // Empty list [] if none
      }}
    }}
    // Repeat this structure for each processed resume, using its correct identifier
  ]
}}

Your JSON Response:
"""

        # --- Call AI Service ---
        # Ensure GITHUB_TOKEN is correctly loaded
        if not GITHUB_TOKEN or GITHUB_TOKEN == "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": # Check your specific placeholder
             print("CRITICAL WARNING: Using a placeholder/fallback AI API Key. Ensure MISTRAL_API_KEY is set.")
             # return Response({'error': 'AI service API Key is not configured.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR) # Optional: block in prod

        endpoint = "https://models.github.ai/inference" # Or your specific endpoint
        model_name = "mistral-ai/mistral-small-2503" # Or your preferred model

        client = Mistral(api_key=GITHUB_TOKEN, server_url=endpoint)
        analysis_json = None # To store the parsed AI response

        try:
            print(f"DEBUG: Sending request to AI API ({model_name}) for recruiter analysis ({len(processed_resumes_data)} resumes)...")
            response = client.chat.complete(
                model=model_name,
                messages=[
                    SystemMessage(content="You are an expert AI hiring assistant comparing resumes to job descriptions accurately and objectively, providing structured JSON output."),
                    UserMessage(content=prompt),
                ],
                temperature=0.4, # Lower temp for more predictable structured output
                max_tokens=4000, # Adjust based on expected output size
                top_p=1.0
                # Consider adding response_format={"type": "json_object"} if supported
            )
            print("DEBUG: Received AI response.")

            raw_ai_text = response.choices[0].message.content
            analysis_json = self.clean_ai_json_response(raw_ai_text) # Use the cleaning function

            if analysis_json is None:
                print("ERROR: Failed to parse JSON from AI response after cleaning.")
                # Log the raw text for debugging (be careful with sensitive data in logs)
                print(f"DEBUG: Raw AI text that failed parsing (first 500 chars):\n---\n{raw_ai_text[:500]}\n---")
                return Response({
                    'error': 'The AI response could not be processed into the expected format.',
                    'details': 'The analysis service returned data that was not valid JSON.'
                 }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # --- Validate Parsed JSON Structure ---
            required_top_level = ['job_description_summary', 'comparative_analysis', 'ranking', 'candidate_analysis']
            if not all(key in analysis_json for key in required_top_level):
                 missing_keys = [key for key in required_top_level if key not in analysis_json]
                 print(f"ERROR: Parsed JSON missing top-level keys: {missing_keys}")
                 return Response({'error': f'AI analysis is incomplete. Missing sections: {", ".join(missing_keys)}.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            if not isinstance(analysis_json.get('candidate_analysis'), list): # Check if it's a list
                 print("ERROR: 'candidate_analysis' is missing or not a list.")
                 return Response({'error': 'AI analysis is incomplete. Candidate analysis data is missing or invalid.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Optional: More detailed validation of candidate_analysis structure if needed
            # ...

            print("DEBUG: Successfully parsed and validated AI analysis JSON structure.")

            # --- DATABASE SAVING LOGIC (Inside the 'try' block after successful AI call and parsing) ---
            try:
                print("DEBUG: Attempting to save analysis results to database.")
                current_user = request.user # User is guaranteed to be authenticated

                # 1. Create the main RecruiterAnalysis record
                #    We store the key AI outputs directly here.
                recruiter_analysis_obj = RecruiterAnalysis.objects.create(
                    user=current_user,
                    # You might create/link a JobRequirement object here if you parsed one,
                    # otherwise just store the snapshot.
                    # job_requirement=None, # Example: Link later if needed
                    job_description_snapshot=job_description, # Store the JD used
                    # Store the relevant JSON parts from the AI response
                    comparative_analysis=analysis_json.get('comparative_analysis', ''), # Get safely
                    ranking=analysis_json.get('ranking', []), # Get safely
                    candidate_analysis_details=analysis_json.get('candidate_analysis', []), # Store the full details list
                    status='completed' # Mark as completed since AI call succeeded
                )
                print(f"DEBUG: Created RecruiterAnalysis record with ID: {recruiter_analysis_obj.id}")

                # 2. Create CandidateResume records for each successfully processed resume
                #    Match AI results back to the original file objects using the identifier.
                ai_results_map = {res['resume_identifier']: res for res in analysis_json.get('candidate_analysis', [])}

                saved_candidate_count = 0
                # Iterate through the data we processed initially (file_obj, identifier, text)
                for file_obj, identifier, extracted_text in processed_resumes_data:
                    # Check if the AI provided analysis for this identifier
                    if identifier in ai_results_map:
                        # Create CandidateResume linked to the RecruiterAnalysis obj
                        CandidateResume.objects.create(
                            recruiter_analysis=recruiter_analysis_obj,
                            resume_identifier=identifier,
                            file=file_obj, # *** This saves the actual uploaded file ***
                            extracted_text=extracted_text,
                            status='analyzed' # Mark as analyzed since AI processed it
                        )
                        saved_candidate_count += 1
                    else:
                        # This case should be rare if the prompt is correct and AI follows instructions
                        print(f"WARN: AI analysis result missing for identifier '{identifier}'. Skipping CandidateResume creation for this file.")

                print(f"DEBUG: Saved {saved_candidate_count} CandidateResume records linked to RecruiterAnalysis {recruiter_analysis_obj.id}.")
                # Optional: Check if counts match
                if saved_candidate_count != len(processed_resumes_data):
                    print(f"WARN: Mismatch! Processed {len(processed_resumes_data)} resumes but saved {saved_candidate_count} CandidateResume records.")

            except IntegrityError as db_int_err:
                print(f"ERROR: Database integrity error during saving analysis: {db_int_err}")
                traceback.print_exc()
                # Transaction automatically rolls back. Inform the user.
                return Response({'error': 'Failed to save analysis results due to a database conflict.', 'details': str(db_int_err)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            except Exception as db_err:
                print(f"ERROR: Failed to save analysis results to database: {db_err}")
                traceback.print_exc()
                # Transaction automatically rolls back.
                return Response({'error': 'Failed to save analysis results to the database.', 'details': str(db_err)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            # --- END DATABASE SAVING LOGIC ---

            # --- Final Response Preparation ---
            # Include any non-fatal processing warnings in the response if desired
            if processing_errors:
                analysis_json['processing_warnings'] = processing_errors

            # --- Return the AI analysis JSON to the frontend ---
            # The frontend uses this JSON to display the results. Saving happened successfully.
            print("DEBUG: Analysis and database saving complete. Returning analysis results to frontend.")
            return Response(analysis_json, status=status.HTTP_200_OK)

        except requests.exceptions.RequestException as api_error:
            print(f"ERROR: Error calling AI API: {api_error}")
            # Log more details for server logs if needed
            error_details_for_log = f"Status: {getattr(api_error.response, 'status_code', 'N/A')}, Body: {getattr(api_error.response, 'text', 'N/A')[:500]}..." if hasattr(api_error, 'response') and api_error.response is not None else str(api_error)
            print(f"DEBUG: AI API Request Error Details: {error_details_for_log}")
            return Response({'error': 'Could not connect to the AI analysis service. Please try again later.'}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        except Exception as e:
            # Catch-all for other unexpected errors during AI call or response handling
            print(f"ERROR: Unexpected error during AI processing or response handling: {e}")
            traceback.print_exc()
            return Response({'error': 'An unexpected error occurred during analysis.', 'details': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


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


class ChatMessagesView(APIView):
    permission_classes = [AllowAny] # Or IsAuthenticated

    def get(self, request, format=None):
        # ... (GET method implementation - likely okay) ...
        resume_id = request.query_params.get("resume_id")
        if not resume_id: return Response({"error": "Missing resume_id"}, status=status.HTTP_400_BAD_REQUEST)
        try: resume_obj = Resume.objects.get(id=resume_id)
        except Resume.DoesNotExist: return Response({"error": "Resume not found"}, status=status.HTTP_404_NOT_FOUND)
        except ValueError: return Response({"error": "Invalid resume_id format."}, status=status.HTTP_400_BAD_REQUEST)
        messages = ChatMessage.objects.filter(resume=resume_obj).order_by("created_at")
        serializer = ChatMessageSerializer(messages, many=True)
        return Response(serializer.data)


    def post(self, request, format=None):
        """
        POST: Save a new chat message.
        Expects JSON: resume_id, message, sender ('user' or 'ai')
        """
        resume_id = request.data.get("resume_id")
        message_content = request.data.get("message")
        sender = request.data.get("sender")

        if not resume_id or not message_content or not sender:
            return Response({"error": "Missing resume_id, message, or sender."}, status=status.HTTP_400_BAD_REQUEST)

        # --- THIS LINE NEEDS MESSAGE_SENDER_CHOICES ---
        # Now it should be defined because of the import
        if sender not in dict(MESSAGE_SENDER_CHOICES):
             return Response({"error": f"Invalid sender type: '{sender}'."}, status=status.HTTP_400_BAD_REQUEST)
        # --- END CHECK ---

        try:
            resume_obj = Resume.objects.get(id=resume_id)

            # --- Optional: Extract text if missing ---
            if not resume_obj.extracted_text:
                print(f"DEBUG: ChatMessagesView - Text missing for Resume {resume_id}. Attempting extraction.")
                # ... (Extraction logic using extracted_text field - same as before) ...
                try:
                    if resume_obj.file and hasattr(resume_obj.file, 'path') and os.path.exists(resume_obj.file.path):
                         with open(resume_obj.file.path, 'rb') as f:
                             reader = PdfReader(f)
                             extracted = "".join(page.extract_text() + "\n" for page in reader.pages if page.extract_text())
                             resume_obj.extracted_text = extracted.strip()
                             resume_obj.save(update_fields=['extracted_text'])
                             print(f"DEBUG: ChatMessagesView - Text extracted/saved.")
                    else: print(f"Warning: ChatMessagesView - Source file missing.")
                except Exception as extraction_err: print(f"ERROR extracting text: {extraction_err}")
            # --- End Optional Extraction ---

            chat_message = ChatMessage.objects.create(
                resume=resume_obj,
                user=request.user if request.user.is_authenticated else None,
                sender=sender,
                message=message_content
            )
            serializer = ChatMessageSerializer(chat_message)
            return Response(serializer.data, status=status.HTTP_201_CREATED)

        except Resume.DoesNotExist: return Response({"error": "Resume not found."}, status=status.HTTP_404_NOT_FOUND)
        except ValueError: return Response({"error": "Invalid resume_id format."}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            print(f"ERROR saving chat message: {e}"); traceback.print_exc()
            return Response({"error": "Failed to save chat message.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)




def get_user_role(user: User) -> str:
    """Determines user role based on group membership."""
    if user.groups.filter(name='Recruiter').exists():
        return 'recruiter'
    # You might want to explicitly check for 'Job Seeker' too
    elif user.groups.filter(name='Job Seeker').exists():
         return 'jobseeker'
    # Fallback or handle users not in either group if necessary
    return 'jobseeker' # Default assumption if not Recruiter




class UploadResumeView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, format=None):
        file = request.FILES.get('file')
        validate_only_str = str(request.data.get('validate_only', 'false')).lower()
        validate_only = validate_only_str == 'true'

        # ... (file validation, text extraction, validation - same as before) ...
        if not file: return Response(...)
        if not file.name.lower().endswith('.pdf'): return Response(...)

        validator = ResumeValidator()
        try:
            file.seek(0)
            extracted_text = validator.extract_text(file)
            file.seek(0)
            is_valid, results = validator.is_resume(extracted_text)
            print(f"DEBUG: Validation result: is_valid={is_valid}, service={results.get('service_used')}")

            if validate_only:
                 return Response({ ... }, status=status.HTTP_200_OK)

            if not is_valid:
                 return Response({ ... }, status=status.HTTP_400_BAD_REQUEST)

            # --- Proceed with saving ---
            authenticated_user = request.user
            print(f"DEBUG: Validation passed. Proceeding with saving the resume for user: {authenticated_user.username}")

            current_role = get_user_role(authenticated_user)

            # --- THIS LINE NEEDS USER_ROLE_CHOICES ---
            # Now USER_ROLE_CHOICES should be defined because of the import
            if current_role not in dict(USER_ROLE_CHOICES):
                 print(f"WARNING: User {authenticated_user.username} has an unrecognized role '{current_role}'. Defaulting resume role to 'jobseeker'.")
                 current_role_for_resume = 'jobseeker'
            else:
                 current_role_for_resume = current_role
            # --- END CHECK ---

            print(f"DEBUG: Saving resume with role: {current_role_for_resume}")

            resume = Resume(
                user=authenticated_user,
                uploaded_as_role=current_role_for_resume,
                file=file,
                extracted_text=extracted_text,
                title=os.path.splitext(file.name)[0]
            )
            resume.save()
            print(f"DEBUG: Resume saved successfully with ID: {resume.id}, Role: {resume.uploaded_as_role}")

            serializer = ResumeSerializer(resume)
            return Response(serializer.data, status=status.HTTP_201_CREATED)

        except Exception as e:
            # ... (Consolidated Error Handling - same as before) ...
            print(f"ERROR during resume upload for file '{file.name}': {e}")
            traceback.print_exc()
            # Determine error_message, details, status_code based on exception type
            # ...
            error_message = "An unexpected error occurred during resume upload."
            details = str(e)
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            return Response({"error": error_message, "details": details}, status=status_code)

class UpdateUserRoleView(APIView):
    permission_classes = [IsAuthenticated]

    @transaction.atomic # Ensure group changes are atomic
    def post(self, request, *args, **kwargs):
        user = request.user
        new_role_name = request.data.get('role') # Expect 'Job Seeker' or 'Recruiter'

        if not new_role_name or new_role_name not in ['Job Seeker', 'Recruiter']:
            return Response(
                {"error": "Invalid or missing 'role'. Must be 'Job Seeker' or 'Recruiter'."},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Get the target group
            target_group = Group.objects.get(name=new_role_name)

            # Determine the group to remove (the opposite role)
            current_role_name = get_user_role(user).capitalize() # e.g., 'Jobseeker' -> 'Job Seeker'
            if new_role_name == 'Job Seeker':
                group_to_remove_name = 'Recruiter'
            else: # new_role_name == 'Recruiter'
                group_to_remove_name = 'Job Seeker'

            # Remove from the old group(s) if they exist and user is in them
            try:
                old_group = Group.objects.get(name=group_to_remove_name)
                if user.groups.filter(name=group_to_remove_name).exists():
                    user.groups.remove(old_group)
                    print(f"DEBUG: Removed user {user.username} from group '{group_to_remove_name}'")
            except Group.DoesNotExist:
                print(f"DEBUG: Group '{group_to_remove_name}' does not exist, skipping removal.")
                pass # Group doesn't exist, nothing to remove

            # Add to the new group if not already in it
            if not user.groups.filter(name=new_role_name).exists():
                 user.groups.add(target_group)
                 print(f"DEBUG: Added user {user.username} to group '{new_role_name}'")
            else:
                 print(f"DEBUG: User {user.username} already in group '{new_role_name}'")


            # Optionally update a profile field if you have one
            # user.profile.role = new_role_name.lower().replace(' ','') # Example
            # user.profile.save()

            return Response({
                "message": f"User role successfully updated to {new_role_name}.",
                "user": { # Return updated user info
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "role": get_user_role(user) # Re-fetch role after update
                }
            }, status=status.HTTP_200_OK)

        except Group.DoesNotExist:
            print(f"ERROR: Target group '{new_role_name}' not found in database.")
            # Transaction will roll back
            return Response(
                {"error": f"Server configuration error: Role group '{new_role_name}' not found."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        except Exception as e:
            print(f"ERROR updating user role for {user.username}: {e}")
            traceback.print_exc()
            # Transaction will roll back
            return Response(
                {"error": "An unexpected error occurred while updating the user role."},
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

try:
    from mistralai import Mistral, UserMessage, SystemMessage
except ImportError: Mistral = None; UserMessage=None; SystemMessage=None; print("MistralAI missing")


import logging
import os
import traceback
from django.db import transaction, IntegrityError
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from mistralai.models import SystemMessage, UserMessage

GITHUB_TOKEN="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"

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AnalyzeResumeView(APIView):
    permission_classes = [AllowAny]  # Or [IsAuthenticated]

    def analyze_resume_github(self, resume_text):
        """Calls Mistral/GitHub endpoint for analysis."""
        if not Mistral:
            logger.error("MistralAI library not configured.")
            return None

        
        if not GITHUB_TOKEN:
            logger.error("GITHUB_TOKEN not configured.")
            return None

        try:
            logger.debug("Initializing Mistral client.")
            client = Mistral(api_key="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", server_url="https://models.github.ai/inference")
            prompt_content = f"""
Analyze the following resume and return ONLY valid JSON following the structure:
{{
  "scores": {{"skills": integer, "experience": integer, "education": integer, "overall": integer}},
  "key_insights": ["string", ... (exactly 10)],
  "improvement_suggestions": ["string", ... (exactly 10)]
}}

Resume Text:
---
{resume_text}
---
Your JSON Response:"""
            logger.debug("Sending request to Mistral AI.")
            response = client.chat.complete(
                model="mistral-ai/mistral-small-2503",
                messages=[
                    SystemMessage(content="You are an expert Resume ATS analyzer providing structured JSON output."),
                    UserMessage(content=prompt_content)
                ],
                temperature=0.5,
                max_tokens=2000,
                top_p=1.0
            )
            logger.debug("AI Analysis API response received.")
            return response.choices[0].message.content if response.choices and response.choices[0].message else None
        except Exception as e:
            logger.error(f"ERROR during AI analysis call: {e}")
            traceback.print_exc()
            return None

    @transaction.atomic
    def post(self, request, format=None):
        logger.debug("POST request received.")
        resume_id = request.data.get("resume_id")
        if not resume_id:
            logger.error("No resume_id provided.")
            return Response({"error": "No resume_id provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            logger.debug(f"Fetching resume with ID: {resume_id}.")
            resume = Resume.objects.get(id=resume_id)
        except Resume.DoesNotExist:
            logger.error(f"Resume not found for ID: {resume_id}.")
            return Response({"error": "Resume not found"}, status=status.HTTP_404_NOT_FOUND)

        # --- Use 'extracted_text' field ---
        resume_text_content = resume.extracted_text
        if not resume_text_content:
            logger.debug(f"Text not found for Resume {resume_id}. Attempting extraction.")
            try:
                if resume.file and hasattr(resume.file, 'path') and os.path.exists(resume.file.path):
                    validator = ResumeValidator()  # Assumes validator is defined/imported
                    logger.debug("Extracting text from resume file.")
                    resume_text_content = validator.extract_text(resume.file)  # Extract text
                    if resume_text_content:
                        resume.extracted_text = resume_text_content  # Save to correct field
                        resume.save(update_fields=['extracted_text'])
                        logger.debug(f"Text extracted and saved for Resume {resume_id}")
                    else:
                        logger.error("Could not extract text from PDF.")
                        return Response({"error": "Could not extract text from PDF."}, status=status.HTTP_400_BAD_REQUEST)
                else:
                    logger.error("Resume text missing and source file unavailable.")
                    return Response({"error": "Resume text missing and source file unavailable."}, status=status.HTTP_400_BAD_REQUEST)
            except (ValueError, ImportError, Exception) as extraction_err:  # Catch errors from extract_text
                logger.error(f"Failed to process resume file: {extraction_err}")
                traceback.print_exc()
                return Response({"error": f"Failed to process resume file: {extraction_err}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        if not resume_text_content:
            logger.error("Resume text is empty.")
            return Response({"error": "Resume text is empty."}, status=status.HTTP_400_BAD_REQUEST)
        # --- End Text Access Fix ---

        logger.debug("Calling analyze_resume_github method.")
        analysis_text = self.analyze_resume_github(resume_text_content)
        if not analysis_text:
            logger.error("Analysis method failed.")
            return Response({"error": "Analysis method failed."}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        analysis_result_json = None
        try:
            logger.debug("Cleaning and parsing JSON response.")
            cleaned_text = clean_json_string(analysis_text)
            analysis_result_json = json.loads(cleaned_text)
            logger.debug("Successfully parsed AI JSON response.")
        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to parse AI JSON: {json_err}\nRaw Text: {analysis_text[:500]}...")
            return Response({"error": "AI response format error."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except Exception as parse_err:
            logger.error(f"Unexpected JSON parsing error: {parse_err}")
            traceback.print_exc()
            return Response({"error": "Internal error processing AI response."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        if analysis_result_json:
            try:
                scores = analysis_result_json.get('scores', {})
                insights = analysis_result_json.get('key_insights', [])
                suggestions = analysis_result_json.get('improvement_suggestions', [])
                analysis_obj, created = Analysis.objects.update_or_create(
                    resume=resume,
                    defaults={
                        'user_role': get_user_role(request.user) if request.user.is_authenticated else 'jobseeker',
                        'skills_score': scores.get('skills'),
                        'experience_score': scores.get('experience'),
                        'education_score': scores.get('education'),
                        'overall_score': scores.get('overall'),
                        'key_insights': insights,
                        'improvement_suggestions': suggestions
                    }
                )
                logger.debug(f"Analysis record {'created' if created else 'updated'} (ID: {analysis_obj.id}) for Resume {resume.id}")
                return Response(analysis_result_json, status=status.HTTP_200_OK)  # Return parsed JSON
            except (IntegrityError, TypeError, Exception) as db_error:
                logger.error(f"ERROR saving analysis to DB: {db_error}")
                traceback.print_exc()
                return Response({"error": "Database error saving analysis."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            logger.error("Analysis failed to produce results.")
            return Response({"error": "Analysis failed to produce results."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
       


class ChatView(APIView):
    # Decide on permissions (AllowAny allows unauth chat, IsAuthenticated requires login)
    # If IsAuthenticated, saving messages will always associate with request.user
    permission_classes = [AllowAny]

    @transaction.atomic # Good practice if saving multiple objects
    def post(self, request, format=None):
        resume_id = request.data.get("resume_id")
        user_message_content = request.data.get("message") # User's input message

        if not resume_id or not user_message_content:
            return Response({"error": "Missing resume_id or message"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            resume_obj = Resume.objects.get(id=resume_id)
            # --- Get resume text (using extracted_text field) ---
            resume_text_content = resume_obj.extracted_text
            if not resume_text_content:
                print(f"DEBUG: ChatView - Text not found for Resume {resume_id}. Attempting extraction.")
                try:
                    if resume_obj.file and hasattr(resume_obj.file, 'path') and os.path.exists(resume_obj.file.path):
                        validator = ResumeValidator() # Instantiate validator for extraction
                        extracted_text = validator.extract_text(resume_obj.file)
                        if extracted_text:
                             resume_obj.extracted_text = extracted_text
                             resume_obj.save(update_fields=['extracted_text'])
                             resume_text_content = extracted_text # Use the newly extracted text
                             print(f"DEBUG: ChatView - Text extracted and saved for Resume {resume_id}")
                        else:
                             print(f"Warning: ChatView - Extraction yielded empty text for Resume {resume_id}")
                             resume_text_content = "" # Proceed with empty context
                    else:
                         print(f"Warning: ChatView - Source file missing or invalid path for Resume {resume_id}")
                         resume_text_content = "" # Proceed with empty context
                except (ValueError, ImportError, Exception) as extraction_err:
                     print(f"ERROR extracting text in ChatView for {resume_id}: {extraction_err}")
                     traceback.print_exc()
                     # Decide how to handle: error out or proceed with empty context?
                     # For now, proceed with empty context but log error
                     resume_text_content = ""

            # Use placeholder if text is empty or unavailable after attempts
            if not resume_text_content:
                 resume_context_for_prompt = "[Resume content not available or could not be processed]"
                 print(f"DEBUG: ChatView - Proceeding without resume text context for Resume {resume_id}.")
            else:
                 resume_context_for_prompt = resume_text_content
                 print(f"DEBUG: ChatView - Using resume text content for prompt.")
            # --- End text fetching ---

            # --- Save USER Message (if authenticated) ---
            current_user = request.user if request.user.is_authenticated else None
            if current_user:
                try:
                    ChatMessage.objects.create(
                        resume=resume_obj,
                        user=current_user,
                        sender='user', # Mark as from user
                        message=user_message_content # Save user's original message
                    )
                    print(f"DEBUG: Saved USER chat message for user {current_user.username}, resume {resume_id}")
                except Exception as save_err:
                    print(f"ERROR: Failed to save USER chat message to DB: {save_err}")
                    # Log error but proceed with AI call if possible
            # --- End Save User Message ---

            # --- Construct Chat Prompt ---
            chat_prompt = (
                 "You are an expert ATS resume advisor. Answer the user's question. "
                 "If resume content is available below, reference specific details from it (skills, education, experience, etc.). "
                 "Do not provide generic advice if specific details are present. Tailor your answer based on the available information. "
                 "If the resume content is marked as unavailable or lacks sufficient details for the question, mention that explicitly. Keep your answer concise, ideally under 100 words.\n\n"
                 "Resume Content:\n---\n" + resume_context_for_prompt + "\n---\n\n" +
                 "User: " + user_message_content + "\n" +
                 "AI:"
            )
            # --- End Prompt Construction ---

            # --- AI Call ---
            final_reply = "Sorry, the AI service encountered an issue." # Default error reply
            try:
                token = "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"; endpoint = "https://models.github.ai/inference" ; model_name = "mistral-ai/mistral-small-2503"
                if not Mistral or not token or token == "YOUR_FALLBACK_MISTRAL_KEY":
                     raise ConnectionError("AI Service (Mistral) not configured or key missing.")

                client = Mistral(api_key=token, server_url=endpoint)
                ai_response = client.chat.complete(
                    model=model_name,
                    messages=[ SystemMessage(content="You are a helpful Resume ATS assistant."), UserMessage(content=chat_prompt)],
                    temperature=0.7, max_tokens=250, top_p=1.0
                 )
                print(f"DEBUG: Chat AI response received.")
                if ai_response.choices and ai_response.choices[0].message:
                    reply = ai_response.choices[0].message.content
                    parts = reply.split("AI:") # Simple cleanup
                    final_reply = parts[-1].strip() if len(parts) > 1 else reply.strip()
                    if not final_reply: # Handle case where AI gives empty response after "AI:"
                         final_reply = "The AI provided an empty response."
                else:
                     final_reply = "AI did not provide a valid response structure."

            except ConnectionError as conn_err: # Catch config errors
                 print(f"ERROR: {conn_err}")
                 # Don't expose config details to user
                 final_reply = "AI service is currently unavailable due to configuration issues."
                 # Optionally return 503 here if preferred
                 # return Response({"error": final_reply}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
            except Exception as ai_error:
                 print(f"ERROR during Chat AI call: {ai_error}"); traceback.print_exc()
                 final_reply = "Sorry, there was an error communicating with the AI assistant."
                 # Optionally return 503 here if preferred
                 # return Response({"error": "Chat AI service failed.", "details": str(ai_error)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
            # --- End AI Call ---

            # --- Save AI Message (if user is authenticated) ---
            # Save even if the reply is an error message generated above,
            # unless it was a critical config error maybe.
            if current_user:
                try:
                    ChatMessage.objects.create(
                        resume=resume_obj,
                        user=current_user, # Still associate with the user who initiated
                        sender='ai', # Mark as from AI
                        message=final_reply # Save the AI's reply (or error message)
                    )
                    print(f"DEBUG: Saved AI chat message for user {current_user.username}, resume {resume_id}")
                except Exception as save_err:
                     print(f"ERROR: Failed to save AI chat message to DB: {save_err}")
                     # Log error, but return the reply anyway
            # --- End Save AI Message ---

            # --- Return the AI reply (or error message) to the frontend ---
            return Response({"reply": final_reply}, status=status.HTTP_200_OK)

        except Resume.DoesNotExist:
            return Response({"error": "Resume not found"}, status=status.HTTP_404_NOT_FOUND)
        except ValueError: # Handle non-integer resume_id if possible
             return Response({"error": "Invalid resume_id format."}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            # Catch-all for unexpected errors during setup/DB access
            print(f"ERROR processing chat request view: {e}")
            traceback.print_exc()
            return Response({"error": "Chat processing failed unexpectedly.", "details": str(e)},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        

class SignupView(APIView):
    permission_classes = [AllowAny]

    # Wrap in transaction to ensure user and group assignment are atomic
    @transaction.atomic
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

        # Validate email format (optional but recommended)
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
             return Response({"error": "Invalid email format."}, status=status.HTTP_400_BAD_REQUEST)

        # Check if username or email already exists
        if User.objects.filter(username=username).exists():
            return Response(
                {"error": "Username already exists."},
                status=status.HTTP_409_CONFLICT
            )
        if User.objects.filter(email=email).exists():
            return Response(
                {"error": "Email address already registered."},
                status=status.HTTP_409_CONFLICT
            )

        try:
            # 1. Create the user
            user = User.objects.create_user(username=username, email=email, password=password)
            print(f"User created successfully: {user.username}")

            # 2. Assign to 'Job Seeker' group (DEFAULT ROLE)
            try:
                job_seeker_group = Group.objects.get(name='Job Seeker')
                user.groups.add(job_seeker_group)
                print(f"Assigned default role 'Job Seeker' to user {user.username}")
                assigned_role = 'Job Seeker'
            except Group.DoesNotExist:
                # Critical error if the default group doesn't exist
                print(f"CRITICAL ERROR: 'Job Seeker' group not found in database. Cannot assign default role.")
                # Rollback transaction explicitly or let the exception bubble up
                # For now, let's return an error, transaction.atomic will handle rollback
                raise Exception("Server configuration error: Default user role group missing.")
                # Alternatively, allow signup without role, but log a severe warning:
                # assigned_role = None # Indicate role assignment failed

            # 3. Generate token for the new user
            token, created = Token.objects.get_or_create(user=user)
            print(f"Token {'created' if created else 'retrieved'} for new user.")

            # 4. Return token and user info (including the assigned role)
            return Response({
                "token": token.key,
                "user": {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "role": assigned_role # Return the role assigned
                    }
            }, status=status.HTTP_201_CREATED)

        except IntegrityError as e: # Catch potential race conditions during user creation
            print("Signup Integrity Error:", str(e))
            # Transaction will roll back
            return Response(
                {"error": "Signup failed due to a database conflict (username/email likely taken)."},
                status=status.HTTP_409_CONFLICT
            )
        except Exception as e:
            print("Signup error:", str(e))
            traceback.print_exc()
            # Transaction will roll back
            # Return specific error if it's the Group.DoesNotExist we raised
            if "Default user role group missing" in str(e):
                 return Response(
                    {"error": "Signup failed due to a server configuration issue."},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                 )
            return Response(
                {"error": "Signup failed due to an unexpected server error.", "details": str(e)},
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
            print(f"Login failed for username: {username}")
            return Response(
                {"error": "Invalid credentials."},
                status=status.HTTP_401_UNAUTHORIZED)

        token, created = Token.objects.get_or_create(user=user)
        print(f"Login successful for user: {user.username}. Token {'created' if created else 'retrieved'}.")

        # --- Determine user role (Example using Groups) ---
        user_role = 'Unknown' # Default if no group found
        user_groups = user.groups.all()
        if user_groups:
            # Prioritize 'Recruiter' if user is in multiple groups for some reason
            if user_groups.filter(name='Recruiter').exists():
                user_role = 'Recruiter'
            elif user_groups.filter(name='Job Seeker').exists():
                user_role = 'Job Seeker'
            else:
                user_role = user_groups.first().name # Fallback to the first group name
        # --- End Role Determination ---

        return Response({
            "token": token.key,
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "role": user_role # Return the determined role
                }
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


# views.py

# ... other imports ...

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_conversations(request):
    """
    Returns a summary list of conversations (represented by resumes)
    for the authenticated user.
    """
    try:
        # --- FIX THE FIELD NAME FOR ORDERING ---
        resumes = Resume.objects.filter(user=request.user).order_by('-upload_date') # Changed from '-uploaded_at'
        # --- END FIX ---

        conversations = []
        for resume in resumes:
            # Use title or generate a fallback name
            resume_name = resume.title if resume.title else f"Resume {resume.id} ({resume.upload_date.strftime('%Y-%m-%d')})"
            conversations.append({
                "resume_id": resume.id,
                "resume_name": resume_name
            })
        return Response(conversations)
    except Exception as e:
        print(f"Error fetching user conversations for {request.user.username}: {e}")
        traceback.print_exc() # Log the full error
        return Response(
            {"error": "Failed to retrieve conversation history."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


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

# views.py

# --- Ensure all necessary imports are present ---
import os
import io
import json
import re
import traceback
from django.utils import timezone
from django.conf import settings
from django.http import JsonResponse, FileResponse, HttpResponse
from django.contrib.auth.models import User, Group
from django.db import IntegrityError, transaction
from rest_framework.views import APIView
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework import status
import requests
import PyPDF2
from PyPDF2 import PdfReader, errors as PyPDF2Errors

try:
    from mistralai import Mistral, UserMessage, SystemMessage
except ImportError:
    Mistral = None; UserMessage=None; SystemMessage=None; print("MistralAI missing")

# Local Imports (ensure these are correct)
from .models import Resume # Ensure Resume is imported
# Import helpers if defined elsewhere
# from .utils import clean_json_string

# --- Constants ---
GITHUB_TOKEN = os.getenv("MISTRAL_API_KEY", "YOUR_FALLBACK_MISTRAL_KEY")

# --- Paste helper functions if defined here ---
# Example:
def clean_json_string(json_str):
    if not isinstance(json_str, str): return json_str
    cleaned = json_str.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
    cleaned = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', cleaned)
    first_brace = cleaned.find('{'); last_brace = cleaned.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        cleaned = cleaned[first_brace : last_brace + 1]
    try:
        pattern_start = r'{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:'; cleaned = re.sub(pattern_start, r'{"\1":', cleaned)
        pattern_mid = r',\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:'; cleaned = re.sub(pattern_mid, r',"\1":', cleaned)
    except Exception as e: print(f"Warning: Regex error during JSON key quoting fix: {e}")
    return cleaned

import logging
import json
import re
import os
import requests
import traceback # For detailed error printing

from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny # Adjust as needed
from rest_framework.response import Response
from django.db import IntegrityError

# --- Model Imports ---
# Adjust the path '.' if your models are elsewhere
from .models import Resume, UpgradedResume

# --- PDF Reader Import ---
try:
    from PyPDF2 import PdfReader
    from PyPDF2 import errors as PyPDF2Errors
except ImportError:
    PdfReader = None
    PyPDF2Errors = None
    print("CRITICAL WARNING: PyPDF2 library not found. PDF processing will be unavailable.")

# --- Environment Variable for API Key ---
# Load API key securely from environment variable
GITHUB_TOKEN = os.environ.get("MISTRAL_API_KEY", "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") # <<< Replace hardcoded fallback in production

# Using standard logging is generally preferred over print
logger = logging.getLogger(__name__)

@api_view(['POST'])
@permission_classes([AllowAny]) # Consider using IsAuthenticated for production
def rewrite_resume(request):
    """
    Endpoint to rewrite a resume using AI.
    - Extracts text from PDF if needed.
    - Calls AI with specific prompt for JSON output ("rewritten_markdown").
    - Performs detailed parsing/extraction of AI response.
    - Saves extracted markdown AND raw AI JSON response to UpgradedResume model.
    - Returns extracted markdown content to frontend under "rewritten_content" key.
    - Includes detailed debugging prints and fallback mock data.
    """
    resume_id = request.data.get('resume_id')
    if not resume_id:
        print("ERROR: Rewrite request missing resume_id")
        logger.error("Rewrite request missing resume_id")
        return Response({'error': 'Resume ID not provided'}, status=status.HTTP_400_BAD_REQUEST)

    print(f"\n--- Starting Rewrite Request for Resume ID: {resume_id} ---")

    # --- Check Library Availability ---
    if not Mistral or not PdfReader:
        error_msg = "Server configuration error: Required libraries (MistralAI or PyPDF2) are missing."
        print(f"ERROR: {error_msg}")
        logger.critical(error_msg)
        return Response({'error': error_msg}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    try:
        # --- Step 1: Get Resume and Original Content ---
        resume = Resume.objects.get(id=resume_id)
        print(f"INFO: Found Resume {resume_id} for user {resume.user.username}")

        original_content = resume.extracted_text # Use 'extracted_text' field from Resume model

        if not original_content or not original_content.strip():
            print(f"INFO: Extracted text empty for Resume {resume_id}. Attempting PDF extraction.")
            try:
                if not resume.file or not hasattr(resume.file, 'path') or not resume.file.path or not os.path.exists(resume.file.path):
                    raise FileNotFoundError(f"Resume file path not found or invalid: {getattr(resume.file, 'path', 'N/A')}")

                print(f"DEBUG: Opening PDF file at: {resume.file.path}")
                with open(resume.file.path, 'rb') as f:
                    pdf_reader = PdfReader(f)
                    if not pdf_reader.pages:
                        print(f"WARNING: PDF for resume {resume_id} has no pages or is unreadable.")
                        raise ValueError("PDF contains no pages or is unreadable.")

                    extracted_text_from_pdf = ""
                    for i, page in enumerate(pdf_reader.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                extracted_text_from_pdf += page_text + "\n"
                            else:
                                print(f"DEBUG: Page {i+1} of PDF {resume_id} yielded no text.")
                        except Exception as page_extract_err:
                            print(f"WARNING: Error extracting text from page {i+1} of PDF {resume_id}: {page_extract_err}")
                            # Continue to next page if one page fails

                original_content = extracted_text_from_pdf.strip() # Use the newly extracted text

                if not original_content:
                    print(f"WARNING: Could not extract any text from PDF for resume {resume_id} after processing pages.")
                    raise ValueError("Could not extract any text from PDF.")

                # Save the extracted text back to the original Resume object
                resume.extracted_text = original_content
                resume.save(update_fields=['extracted_text'])
                print(f"INFO: Successfully extracted text (length: {len(original_content)}) from PDF and saved to resume {resume_id}")

            except FileNotFoundError as fnf_error:
                print(f"ERROR: PDF file not found for resume {resume_id}: {fnf_error}")
                return Response({'error': 'Original resume file not found.', 'details': str(fnf_error)}, status=status.HTTP_404_NOT_FOUND)
            except (ValueError, PyPDF2Errors.PdfReadError, Exception) as extraction_error:
                print(f"ERROR: Failed extracting text from PDF for resume {resume_id}: {extraction_error}")
                traceback.print_exc()
                error_detail = str(extraction_error)
                status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
                if isinstance(extraction_error, PyPDF2Errors.PdfReadError):
                    error_detail = "Could not read the PDF file; it might be corrupted or encrypted."
                    status_code = status.HTTP_400_BAD_REQUEST
                elif isinstance(extraction_error, ValueError):
                    status_code = status.HTTP_400_BAD_REQUEST # Bad PDF content or structure
                return Response({'error': 'Failed to process the original resume PDF.', 'details': error_detail}, status=status_code)

        # Final check: Do we have content now?
        if not original_content or not original_content.strip():
            print(f"ERROR: Original resume content is still empty for resume {resume_id} after all attempts.")
            return Response({'error': 'Original resume content is empty or could not be read.'}, status=status.HTTP_400_BAD_REQUEST)

        print(f"DEBUG: Original content length for AI prompt: {len(original_content)}")

        # --- Step 2: Construct AI Prompt (Using the detailed sample prompt) ---
        prompt = f"""
You are an expert ATS resume writer and formatter. Your task is to rewrite the provided raw resume text to be highly impactful, professional, ATS-optimized, and structured precisely in Markdown format.

Core Instructions:
1. Maintain Information: Preserve ALL original details (names, dates, companies, skills, descriptions, locations, contact details, etc.).
2. Enhance Wording: Improve clarity, use strong action verbs, quantify achievements, and ensure professional language.
3. ATS Optimization: Naturally integrate relevant keywords.
4. Markdown Structure: Format the rewritten resume using the specified Markdown structure below. Use '*' for ALL bullet points. Ensure proper spacing (e.g., '# Heading', '* Bullet').
5. Output Format: Respond ONLY with a valid JSON object containing a single key "rewritten_markdown". The value MUST be a string containing the complete rewritten resume in Markdown, starting with '# Full Name'.
6. Strictness: Do NOT include any introductory text, explanations, or code block markers (like ```json) outside the JSON object itself.

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

---

Your Response (JSON Object Only):
"""

        # --- Step 3: Call AI Service ---
        # Ensure GITHUB_TOKEN is loaded correctly from environment or replace fallback
        if not GITHUB_TOKEN or GITHUB_TOKEN == "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": # Check against your specific placeholder/fallback
            print("CRITICAL WARNING: Using a placeholder/fallback AI API Key. Ensure MISTRAL_API_KEY environment variable is set correctly.")
            # Consider returning an error in production if the key is clearly invalid/missing
            # return Response({'error': 'AI Service API Key is not configured on the server.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Use the endpoint appropriate for your Mistral client setup
        endpoint = "https://models.github.ai/inference" # Or your Azure/other endpoint
        model_name = "mistral-ai/mistral-small-2503" # Or your preferred model

        client = Mistral(api_key=GITHUB_TOKEN, server_url=endpoint)
        extracted_markdown_content = None # Variable for the final markdown string
        ai_response_object = None         # Variable for the raw parsed JSON object from AI

        try:
            print(f"INFO: Sending request to AI API ({model_name} at {endpoint}) for resume {resume_id}")
            response = client.chat.complete(
                model=model_name,
                messages=[
                    SystemMessage(content="You are an expert ATS resume writer and formatter following specific JSON output instructions."),
                    UserMessage(content=prompt)
                ],
                temperature=0.6, # Adjust temperature as needed
                max_tokens=4000, # Ensure sufficient tokens
                top_p=1.0,
                response_format={"type": "json_object"} # Request JSON directly
            )

            # --- Step 4: Parse AI Response (Detailed logic) ---
            if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
                print("ERROR: AI response structure invalid (no choices or message content found).")
                ai_response_object = {"error": "Invalid AI response structure", "details": "No choices or message content found"}
                # Continue to fallback/mock data, don't raise error here yet
            else:
                raw_generated_text = response.choices[0].message.content
                print(f"DEBUG: Raw AI response text received (first 1000 chars):\n---\n{raw_generated_text[:1000]}\n---")

                # Attempt to clean potential code block markers first
                cleaned_text = re.sub(r'^```(?:json)?\s*|\s*```\\$', '', raw_generated_text, flags=re.MULTILINE | re.DOTALL).strip()
                # print(f"DEBUG: Cleaned AI response text (first 1000 chars):\n---\n{cleaned_text[:1000]}\n---") # Optional

                # --- Try parsing the cleaned text as JSON ---
                try:
                    parsed_json = json.loads(cleaned_text)
                    ai_response_object = parsed_json # Store the successfully parsed JSON
                    print("DEBUG: Successfully parsed cleaned text as JSON.")

                    # Check for the required key and valid content
                    if isinstance(parsed_json, dict) and "rewritten_markdown" in parsed_json and isinstance(parsed_json["rewritten_markdown"], str):
                        markdown_candidate = parsed_json["rewritten_markdown"].strip()
                        if markdown_candidate: # Ensure content is not just whitespace
                            extracted_markdown_content = markdown_candidate
                            print("INFO: Successfully extracted content from 'rewritten_markdown' key.")
                        else:
                            print("WARN: 'rewritten_markdown' key found but content is empty.")
                            ai_response_object["parsing_warning"] = "Key 'rewritten_markdown' contained empty string."
                    else:
                        print("WARN: Parsed JSON but 'rewritten_markdown' key missing, not a string, or JSON is not a dict.")
                        ai_response_object["parsing_warning"] = "Key 'rewritten_markdown' missing or invalid type."

                except json.JSONDecodeError as json_err:
                    print(f"WARN: Failed to parse cleaned text directly as JSON: {json_err}. Storing raw text.")
                    ai_response_object = {"error": "Failed to parse AI response as JSON", "raw_text": cleaned_text}
                    # --- Fallback: Attempt direct Markdown extraction ---
                    print("INFO: Attempting direct markdown extraction as fallback.")
                    markdown_markers = ["# ", "## ", "### "]
                    content_lines = cleaned_text.splitlines()
                    start_index = -1
                    for i, line in enumerate(content_lines):
                        trimmed_line = line.strip()
                        # Look for lines starting with #, ##, ### etc.
                        if trimmed_line.startswith('#'):
                            start_index = i
                            print(f"DEBUG: Found potential markdown start at line {i}: '{trimmed_line[:80]}'")
                            break # Assume first # starts the content

                    if start_index != -1:
                        potential_content = "\n".join(content_lines[start_index:]).strip()
                        # Basic validation: Does it contain common sections? More robust check needed?
                        if ("## Summary" in potential_content or "## Experience" in potential_content or "## Skills" in potential_content) and len(potential_content) > 50:
                            extracted_markdown_content = potential_content
                            print("INFO: Extracted markdown content directly based on header markers (fallback).")
                            # Update ai_response_object to indicate fallback success
                            ai_response_object["parsing_info"] = "Markdown extracted via fallback pattern matching."
                        else:
                            print("WARN: Found potential markdown start, but content seemed incomplete or lacked common sections.")
                    else:
                        print("WARN: Could not find clear markdown header markers for direct extraction fallback.")

        except requests.exceptions.RequestException as api_error:
            print(f"ERROR: AI API Request failed: {api_error}")
            traceback.print_exc()
            ai_response_object = {"error": "AI API request failed", "details": str(api_error)}
            # Let it proceed to mock data/saving the error, but maybe return 503?
            # For now, let it fall through to mock data below

        except Exception as e:
            print(f"ERROR: Unexpected error during AI processing or parsing for resume {resume_id}: {e}")
            traceback.print_exc()
            ai_response_object = {"error": "Unexpected processing error", "details": str(e)}
            # Let it proceed to mock data/saving the error

        # --- Step 5: Final Check and Mock Data Fallback ---
        final_content_to_return = None
        message_to_frontend = ""
        status_code_for_response = status.HTTP_200_OK # Default to OK

        if extracted_markdown_content and extracted_markdown_content.strip():
            final_content_to_return = extracted_markdown_content # Use the successfully extracted content
            message_to_frontend = "Resume rewrite processed successfully using AI response."
            print(f"INFO: Processing successful. Final content length: {len(final_content_to_return)}")
        else:
            # --- Apply Mock Data Fallback ---
            print("ERROR: Failed to extract valid markdown content from AI response after all attempts. Using mock data.")
            mock_data = (
                "# JOHN DOE (Mock Data)\n"
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
                "*[Note: This is mock data as the AI service response could not be processed reliably.]*"
            )
            final_content_to_return = mock_data
            message_to_frontend = "AI processing failed; placeholder content is shown. Please review."
            status_code_for_response = status.HTTP_200_OK # Return OK but indicate issue in message
            # Ensure ai_response_object reflects the failure if it wasn't already set
            if ai_response_object is None:
                ai_response_object = {"error": "AI processing failed (reason unknown), mock data used."}
            elif isinstance(ai_response_object, dict):
                ai_response_object["final_status"] = "Mock data used due to processing failure."
            print("INFO: Using mock data as fallback.")

        # --- Step 6: Save to UpgradedResume Model (including raw AI response) ---
        try:
            # Ensure ai_response_object is serializable (it should be if it's dict or None)
            if ai_response_object is not None and not isinstance(ai_response_object, (dict, list, str, int, float, bool)):
                print(f"WARN: ai_response_object is not JSON serializable type ({type(ai_response_object)}), converting to string.")
                ai_response_object_to_save = str(ai_response_object) # Fallback to string representation
            else:
                ai_response_object_to_save = ai_response_object

            upgraded_resume, created = UpgradedResume.objects.update_or_create(
                user=resume.user,          # Link to the same user
                original_resume=resume,   # Link back to the original resume
                defaults={
                    'content_markdown': json.dumps({"rewritten_markdown": final_content_to_return}), # Save the final markdown in JSON format
                    'ai_raw_response': ai_response_object_to_save, # Save the parsed JSON object or error info
                    'title': f"AI Rewritten - {resume.title or f'Resume {resume.id}'}",
                    'status': 'completed' if extracted_markdown_content else 'failed_fallback', # Reflect status
                    'revision_count': 0 # Reset revision count on new rewrite
                }
            )
            print(f"INFO: {'Created' if created else 'Updated'} UpgradedResume record {upgraded_resume.id} (including AI raw response).")

        except (IntegrityError, Exception) as db_error:
            print(f"ERROR: Failed to save rewritten content to UpgradedResume for original resume {resume_id}: {db_error}")
            traceback.print_exc()
            # Return error even if AI worked but DB failed
            return Response(
                {'error': 'Failed to save the rewritten resume data.', 'details': str(db_error)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # --- Step 7: Return Response to Frontend ---
        # CRITICAL: Use the key the frontend expects: "rewritten_content"
        print(f"INFO: Sending response to frontend for resume {resume_id}. Status: {status_code_for_response}")
        return Response({
            'rewritten_content': final_content_to_return, # Send the MARKDOWN content
            'message': message_to_frontend
        }, status=status_code_for_response)

    except Resume.DoesNotExist:
        print(f"ERROR: Resume not found for ID: {resume_id}")
        return Response({'error': 'Resume not found'}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        # Catch-all for unexpected errors early in the process
        print(f"FATAL ERROR in rewrite_resume view setup for resume {resume_id}: {e}")
        traceback.print_exc()
        return Response(
            {'error': 'An unexpected server error occurred processing the rewrite request.', 'details': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    finally:
        print(f"--- Finished Rewrite Request for Resume ID: {resume_id} ---")




# views.py

# ... other imports ...
import logging
import json
import re
import os
import requests
import traceback # For detailed error printing

from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny # Adjust as needed
from rest_framework.response import Response
from django.db import IntegrityError
# --- NEW/Ensure these Model Imports ---
from .models import Resume, UpgradedResume # Import UpgradedResume
from django.db.models import F # Import F expression for atomic updates

# --- PDF Reader Import ---
try:
    from PyPDF2 import PdfReader
    from PyPDF2 import errors as PyPDF2Errors
except ImportError:
    PdfReader = None
    PyPDF2Errors = None
    print("CRITICAL WARNING: PyPDF2 library not found. PDF processing will be unavailable.")

# --- Mistral Import ---
try:
    from mistralai import Mistral, UserMessage, SystemMessage
except ImportError:
    Mistral = None; UserMessage=None; SystemMessage=None; print("MistralAI missing")


# --- Environment Variable for API Key ---
GITHUB_TOKEN = os.environ.get("MISTRAL_API_KEY", "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") # <<< Replace hardcoded fallback in production

logger = logging.getLogger(__name__)

# --- Helper function (ensure it's defined or imported) ---
def clean_json_string(json_str):
    # (Keep your existing clean_json_string function here)
    if not isinstance(json_str, str): return json_str
    cleaned = json_str.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
    cleaned = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', cleaned)
    first_brace = cleaned.find('{'); last_brace = cleaned.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        cleaned = cleaned[first_brace : last_brace + 1]
    try:
        # Basic quoting fix - might need refinement
        pattern_start = r'{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:'; cleaned = re.sub(pattern_start, r'{"\1":', cleaned)
        pattern_mid = r',\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:'; cleaned = re.sub(pattern_mid, r',"\1":', cleaned)
        # Handle trailing commas before closing brace/bracket (simple case)
        cleaned = re.sub(r',\s*(\}|\])', r'\1', cleaned)
    except Exception as e: print(f"Warning: Regex error during JSON key quoting fix: {e}")
    return cleaned


# --- Function to extract markdown from potentially messy AI JSON ---
def extract_markdown_from_ai_response(raw_text, key_name="revised_markdown"):
    """
    Attempts to extract markdown content from AI response.
    Handles direct JSON, JSON within text, or direct markdown.
    Returns the extracted markdown string or None.
    Also returns the parsed JSON object if successful, or a dict with raw text/error.
    """
    print(f"DEBUG: Attempting to extract markdown using key '{key_name}' from raw text (first 500): {raw_text[:500]}...")
    extracted_markdown = None
    parsed_json_object = {"raw_text": raw_text, "parsing_status": "failed_initial"} # Default state

    if not raw_text or not raw_text.strip():
        print("WARN: AI response text is empty.")
        parsed_json_object["parsing_status"] = "failed_empty_response"
        return None, parsed_json_object

    # 1. Clean common markdown code fences
    cleaned_text = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw_text, flags=re.MULTILINE | re.DOTALL).strip()

    # 2. Try parsing the cleaned text directly as JSON
    try:
        parsed_json = json.loads(cleaned_text)
        parsed_json_object = parsed_json # Store the actual parsed object
        if isinstance(parsed_json, dict) and key_name in parsed_json and isinstance(parsed_json[key_name], str):
            markdown_candidate = parsed_json[key_name].strip()
            if markdown_candidate:
                extracted_markdown = markdown_candidate
                parsed_json_object["parsing_status"] = "success_json_key"
                print(f"INFO: Successfully extracted markdown from JSON key '{key_name}'.")
                return extracted_markdown, parsed_json_object
            else:
                parsed_json_object["parsing_warning"] = f"Key '{key_name}' found but content is empty."
                parsed_json_object["parsing_status"] = "failed_empty_content"
                print(f"WARN: JSON key '{key_name}' found but content is empty.")
        else:
            parsed_json_object["parsing_warning"] = f"Key '{key_name}' missing or not a string in parsed JSON."
            parsed_json_object["parsing_status"] = "failed_key_missing_or_invalid"
            print(f"WARN: Parsed JSON but key '{key_name}' missing or invalid type.")

    except json.JSONDecodeError:
        print("DEBUG: Direct JSON parsing failed. Trying fallback methods.")
        parsed_json_object["parsing_status"] = "failed_json_decode"

        # 3. Fallback: Regex to find JSON object within text
        json_pattern = r'\{.*?"' + key_name + r'":\s*".*?".*?\}' # Simplified pattern
        potential_jsons = re.findall(json_pattern, cleaned_text, re.DOTALL)
        if potential_jsons:
            print(f"DEBUG: Found {len(potential_jsons)} potential JSON objects via regex.")
            for potential_json_str in potential_jsons:
                try:
                    parsed_json = json.loads(potential_json_str)
                    parsed_json_object = parsed_json # Store first successfully parsed one
                    if isinstance(parsed_json, dict) and key_name in parsed_json and isinstance(parsed_json[key_name], str):
                        markdown_candidate = parsed_json[key_name].strip()
                        if markdown_candidate:
                            extracted_markdown = markdown_candidate
                            parsed_json_object["parsing_status"] = "success_regex_json_key"
                            print(f"INFO: Successfully extracted markdown via regex JSON key '{key_name}'.")
                            return extracted_markdown, parsed_json_object # Success!
                        else:
                            parsed_json_object["parsing_warning"] = f"Key '{key_name}' (regex) found but content empty."
                            parsed_json_object["parsing_status"] = "failed_regex_empty_content"
                            print(f"WARN: Regex JSON key '{key_name}' found but content is empty.")
                    else:
                         parsed_json_object["parsing_warning"] = f"Key '{key_name}' (regex) missing/invalid."
                         parsed_json_object["parsing_status"] = "failed_regex_key_missing"
                         print(f"WARN: Regex parsed JSON but key '{key_name}' missing or invalid.")
                except json.JSONDecodeError:
                    continue # Try next potential JSON match

        # 4. Fallback: Direct Markdown Extraction (if no JSON worked)
        if not extracted_markdown:
            print("DEBUG: JSON methods failed. Attempting direct markdown extraction.")
            content_lines = cleaned_text.splitlines()
            start_index = -1
            for i, line in enumerate(content_lines):
                trimmed_line = line.strip()
                if trimmed_line.startswith('#'): # Look for main headings
                    start_index = i
                    print(f"DEBUG: Found potential markdown start at line {i}: '{trimmed_line[:80]}'")
                    break
            if start_index != -1:
                potential_content = "\n".join(content_lines[start_index:]).strip()
                # Basic validation: Does it contain common sections?
                if ("## Summary" in potential_content or "## Experience" in potential_content or "## Skills" in potential_content) and len(potential_content) > 50:
                    extracted_markdown = potential_content
                    parsed_json_object["parsing_status"] = "success_direct_markdown"
                    print("INFO: Extracted markdown content directly based on header markers (fallback).")
                    # Update parsed_json_object to reflect this was a fallback
                    parsed_json_object["parsing_info"] = "Markdown extracted via fallback pattern matching."
                    return extracted_markdown, parsed_json_object
                else:
                    parsed_json_object["parsing_status"] = "failed_direct_markdown_validation"
                    print("WARN: Found markdown start, but content seemed incomplete/invalid.")
            else:
                parsed_json_object["parsing_status"] = "failed_direct_markdown_no_header"
                print("WARN: Could not find clear markdown header markers for direct extraction.")

    # If we reach here, all methods failed
    print("ERROR: Failed to extract valid markdown content after all attempts.")
    return None, parsed_json_object


@api_view(['POST'])
@permission_classes([AllowAny]) # Consider IsAuthenticated
def revise_resume(request):
    """
    Endpoint to revise a rewritten resume based on user feedback.
    Saves the result to the UpgradedResume model and increments revision count.
    """
    resume_id = request.data.get('resume_id')
    feedback = request.data.get('feedback')
    current_version_md = request.data.get('current_version') # Expecting markdown string

    print(f"\n--- Starting Revision Request for Resume ID: {resume_id} ---")
    logger.debug(f"Revision request for resume_id: {resume_id}")

    if not resume_id:
        logger.error("Revision request missing resume_id")
        return Response({'error': 'Resume ID not provided'}, status=status.HTTP_400_BAD_REQUEST)
    if not feedback or not feedback.strip():
        logger.error(f"Revision request missing feedback for resume_id: {resume_id}")
        return Response({'error': 'Feedback not provided'}, status=status.HTTP_400_BAD_REQUEST)
    if not current_version_md or not current_version_md.strip():
        logger.error(f"Revision request missing current_version for resume_id: {resume_id}")
        return Response({'error': 'Current resume version not provided'}, status=status.HTTP_400_BAD_REQUEST)

    # --- Check Library Availability ---
    if not Mistral:
        error_msg = "Server configuration error: AI library (MistralAI) is missing."
        print(f"ERROR: {error_msg}")
        logger.critical(error_msg)
        return Response({'error': error_msg}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    try:
        # --- Step 1: Get the Original Resume ---
        # We need the original Resume object to link the UpgradedResume
        resume = Resume.objects.get(id=resume_id)
        print(f"INFO: Found original Resume {resume_id} for user {resume.user.username}")

        # --- Step 2: Construct AI Prompt (similar to before) ---
        # Use the detailed prompt asking for JSON output with "revised_markdown" key
        prompt = f"""
You are an expert ATS resume writer and formatter. Your task is to revise the provided resume based on the user's feedback, while maintaining professional ATS formatting and style.

Core Instructions:
1.  Make Requested Changes: Apply the user's feedback carefully, preserving the overall professional quality.
2.  Maintain Information: Preserve ALL original information that the user doesn't ask to change.
3.  Enhance Wording: Improve clarity, use strong action verbs, quantify achievements, and ensure professional language.
4.  ATS Optimization: Naturally integrate relevant keywords.
5.  Markdown Structure: Format the revised resume using the standard Markdown structure provided below. Use '*' for ALL bullet points. Ensure proper spacing.
6.  Output Format: Respond ONLY with a valid JSON object containing a single key "revised_markdown". The value associated with this key MUST be a string containing the complete, revised resume in Markdown format, starting directly with the '# Full Name' heading.
7.  Strictness: Do NOT include any introductory text, explanations, apologies, code block markers (like ```json), or any text whatsoever before or after the single JSON object in your response.

Markdown Structure Template (for the value of "revised_markdown"):
# [Full Name]
[City, State (if available)] | [Phone Number (if available)] | [Email Address] | [LinkedIn Profile URL (if available, otherwise omit)]

## Summary
[Revised summary text...]

## Skills
* Programming Languages: [Comma-separated list]
* Frameworks & Libraries: [Comma-separated list]
* [...]

## Experience
### [Job Title]
**[Company Name] | [City, State] | [Start Month, Year] – [End Month, Year or Present]
* [Revised responsibility/achievement 1...]
* [Revised responsibility/achievement 2...]

### [Previous Job Title]
**[Previous Company Name] | [...]
* [...]

## Education
### [Degree Name]
**[Institution Name] | [...]
* [Optional bullet...]

## Projects (Include ONLY if distinct)
### [Project Name 1]
* [Description...]

## Certifications (Include ONLY if mentioned)
* [Certification Name...]

---

Current Resume (Markdown):
{current_version_md}

---

User Feedback:
{feedback}

---

Your Response (JSON Object Only):
"""

        # --- Step 3: Call AI Service ---
        # Ensure GITHUB_TOKEN is loaded correctly
        if not GITHUB_TOKEN or GITHUB_TOKEN == "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": # Check your specific fallback
            print("CRITICAL WARNING: Using a placeholder/fallback AI API Key for revision.")
            # Consider returning an error in production

        endpoint = "https://models.github.ai/inference"
        model_name = "mistral-ai/mistral-small-2503" # Or your preferred model

        client = Mistral(api_key=GITHUB_TOKEN, server_url=endpoint)
        final_revised_content = None # Store the successfully extracted markdown
        ai_response_data = None # Store the raw/parsed AI response for the DB

        try:
            print(f"INFO: Sending revision request to AI API ({model_name}) for resume {resume_id}")
            logger.debug(f"Sending revision request to AI API for resume {resume_id}")
            response = client.chat.complete(
                model=model_name,
                messages=[
                    SystemMessage(content="You are an expert ATS resume writer revising content based on feedback and outputting JSON."),
                    UserMessage(content=prompt)
                ],
                temperature=0.6, # Adjust as needed
                max_tokens=4000,
                top_p=1.0,
                response_format={"type": "json_object"} # Request JSON directly
            )

            # --- Step 4: Parse AI Response ---
            if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
                print("ERROR: AI revision response structure invalid.")
                logger.error(f"Invalid AI response structure during revision for resume {resume_id}")
                ai_response_data = {"error": "Invalid AI response structure", "details": "No choices or message content found"}
            else:
                raw_generated_text = response.choices[0].message.content
                # Use the helper function for robust extraction
                extracted_md, parsed_info = extract_markdown_from_ai_response(raw_generated_text, key_name="revised_markdown")
                final_revised_content = extracted_md # Will be None if extraction failed
                ai_response_data = parsed_info # Store the parsed object or error info

        except requests.exceptions.RequestException as api_error:
            print(f"ERROR: AI API Request failed during revision: {api_error}")
            logger.error(f"AI API Request failed during revision for resume {resume_id}: {api_error}", exc_info=True)
            ai_response_data = {"error": "AI API request failed", "details": str(api_error)}

        except Exception as e:
            print(f"ERROR: Unexpected error during AI revision processing for resume {resume_id}: {e}")
            logger.error(f"Unexpected error during AI revision for resume {resume_id}: {e}", exc_info=True)
            traceback.print_exc()
            ai_response_data = {"error": "Unexpected processing error", "details": str(e)}

        # --- Step 5: Check Result and Prepare Final Content ---
        message_to_frontend = ""
        status_code_for_response = status.HTTP_200_OK

        if final_revised_content and final_revised_content.strip():
            # AI revision successful
            message_to_frontend = "Resume revised successfully based on your feedback."
            print(f"INFO: Revision successful. Final content length: {len(final_revised_content)}")
            logger.info(f"Revision successful for resume {resume_id}")
        else:
            # AI revision failed or returned empty content
            print("ERROR: Failed to get valid revised content from AI. Using current version with feedback note.")
            logger.error(f"Failed to get valid revised content from AI for resume {resume_id}. Using placeholder.")
            # Use the previous version + a note as fallback
            final_revised_content = current_version_md + f"\n\n---\n**Revision Attempt Failed**\nFeedback Provided:\n{feedback}\n*AI service did not return a valid revision. Displaying previous version.*"
            message_to_frontend = "AI revision failed; displaying previous version with feedback note. Please try again or refine feedback."
            # Keep status 200 OK, but message indicates failure
            if ai_response_data is None:
                ai_response_data = {"error": "AI revision failed (reason unknown), fallback used."}
            elif isinstance(ai_response_data, dict):
                ai_response_data["final_status"] = "Fallback content used due to processing failure."

        # Optional: Clean up the final markdown content (e.g., excessive newlines)
        if final_revised_content:
            final_revised_content = re.sub(r'\n{3,}', '\n\n', final_revised_content).strip()
            final_revised_content = re.sub(r'(?<=\n)\s*\*\s+', '* ', final_revised_content) # Fix bullet spacing

        # --- Step 6: Save to UpgradedResume Model ---
        try:
            # Ensure ai_response_data is serializable
            if ai_response_data is not None and not isinstance(ai_response_data, (dict, list, str, int, float, bool)):
                print(f"WARN: ai_response_data (type: {type(ai_response_data)}) not JSON serializable, converting to string.")
                ai_response_data_to_save = str(ai_response_data)
            else:
                ai_response_data_to_save = ai_response_data

            upgraded_resume, created = UpgradedResume.objects.update_or_create(
                user=resume.user,          # Link to the same user
                original_resume=resume,   # Link back to the original resume
                defaults={
                    # Store the final markdown content in the expected JSON structure
                    'content_markdown': json.dumps({"rewritten_markdown": final_revised_content}),
                    'ai_raw_response': ai_response_data_to_save, # Save the AI response info
                    'title': f"Revised ({timezone.now().strftime('%Y-%m-%d')}) - {resume.title or f'Resume {resume.id}'}", # Update title
                    'status': 'completed' if (ai_response_data and ai_response_data.get("parsing_status", "").startswith("success")) else 'failed_fallback', # Reflect status
                    # Update revision count using F expression below if updating
                }
            )

            # Increment revision count (only if updating an existing record)
            if not created:
                # Use F expression for atomic increment
                UpgradedResume.objects.filter(pk=upgraded_resume.pk).update(revision_count=F('revision_count') + 1)
                upgraded_resume.refresh_from_db(fields=['revision_count']) # Get the updated value
                print(f"DEBUG: Incremented revision count for UpgradedResume {upgraded_resume.id} to {upgraded_resume.revision_count}")
                logger.debug(f"Incremented revision count for UpgradedResume {upgraded_resume.id} to {upgraded_resume.revision_count}")
            else:
                # If created, revision_count defaults to 0. This is the *first* save via revision.
                # Set count to 1 explicitly if desired for the first revision save.
                upgraded_resume.revision_count = 1
                upgraded_resume.save(update_fields=['revision_count'])
                print(f"DEBUG: Created new UpgradedResume {upgraded_resume.id} via revision, set revision count to 1.")
                logger.debug(f"Created new UpgradedResume {upgraded_resume.id} via revision, set revision count to 1.")

            print(f"INFO: {'Created' if created else 'Updated'} UpgradedResume record {upgraded_resume.id} with revised content.")
            logger.info(f"{'Created' if created else 'Updated'} UpgradedResume record {upgraded_resume.id} for resume {resume_id}")

        except (IntegrityError, Exception) as db_error:
            print(f"ERROR: Failed to save revised content to UpgradedResume for original resume {resume_id}: {db_error}")
            logger.error(f"DB error saving revised UpgradedResume for resume {resume_id}: {db_error}", exc_info=True)
            traceback.print_exc()
            # Return error even if AI worked but DB failed
            return Response(
                {'error': 'Failed to save the revised resume data.', 'details': str(db_error)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # --- Step 7: Return Response to Frontend ---
        # Send the actual markdown content back under the expected key
        print(f"INFO: Sending revision response to frontend for resume {resume_id}. Status: {status_code_for_response}")
        logger.debug(f"Sending revision response for resume {resume_id}")
        return Response({
            'revised_content': final_revised_content, # Send the MARKDOWN string
            'message': message_to_frontend
        }, status=status_code_for_response)

    except Resume.DoesNotExist:
        print(f"ERROR: Original resume not found for ID: {resume_id}")
        logger.error(f"Original resume not found for ID: {resume_id} in revise_resume")
        return Response({'error': 'Original Resume not found'}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        # Catch-all for unexpected errors early in the process
        print(f"FATAL ERROR in revise_resume view setup for resume {resume_id}: {e}")
        logger.critical(f"Fatal error in revise_resume view setup for resume {resume_id}: {e}", exc_info=True)
        traceback.print_exc()
        return Response(
            {'error': 'An unexpected server error occurred processing the revision request.', 'details': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    finally:
        print(f"--- Finished Revision Request for Resume ID: {resume_id} ---")

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
    



def get_user_role(user: User) -> str:
    """Determines user role based on group membership."""
    if not user or not user.is_authenticated:
        return 'guest' # Or handle anonymous users appropriately
    if user.groups.filter(name='Recruiter').exists():
        return 'recruiter'
    elif user.groups.filter(name='Job Seeker').exists():
         return 'jobseeker'
    # Fallback if user is authenticated but not in expected groups
    print(f"Warning: User {user.username} not in 'Recruiter' or 'Job Seeker' group. Defaulting role to 'jobseeker'.")
    return 'jobseeker'



class ResumeValidator:
    """
    Validates if an uploaded PDF file appears to be a resume based on
    text content analysis (keyword and structure checks).
    """
    def extract_text(self, pdf_file_obj):
        """Extracts text from a PDF file object."""
        if not hasattr(PyPDF2, 'PdfReader'):
             raise ImportError("PyPDF2 library is required but seems unavailable.")

        text = ""
        try:
            if hasattr(pdf_file_obj, 'seek'): pdf_file_obj.seek(0)
            pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
            if not pdf_reader.pages:
                print(f"Warning: PDF file {getattr(pdf_file_obj, 'name', 'N/A')} has no pages.")
                return ""
            for page in pdf_reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text: text += page_text + "\n"
                except Exception as page_error:
                    print(f"Warning: Could not extract text from a page in {getattr(pdf_file_obj, 'name', 'N/A')}: {page_error}")
                    continue
            if hasattr(pdf_file_obj, 'seek'): pdf_file_obj.seek(0)
            return text.strip()
        except PyPDF2Errors.PdfReadError as e:
             file_name = getattr(pdf_file_obj, 'name', 'N/A')
             print(f"Error reading PDF {file_name}: {e}")
             if "encrypted" in str(e).lower(): raise ValueError(f"File '{file_name}' is encrypted.")
             else: raise ValueError(f"File '{file_name}' is corrupted or not a valid PDF: {e}")
        except Exception as e:
             file_name = getattr(pdf_file_obj, 'name', 'N/A')
             print(f"Unexpected error during PDF text extraction for {file_name}: {e}")
             traceback.print_exc()
             raise ValueError(f"An unexpected error occurred processing file '{file_name}'.")

    def is_resume_local(self, text):
        """Determines if text is likely a resume using keywords."""
        indicators_found = 0; personal_info_found = 0; confidence = 0.0; is_valid = False; error_message = None
        if not text or not text.strip():
            error_message = "Input text was empty or whitespace."
            return False, { "is_resume": False, "confidence": 0.0, "indicators_found": 0, "personal_info_found": 0, "top_label": "empty", "service_used": "local_keyword_analysis", "error": error_message }
        try:
            text_lower = text.lower()
            resume_indicators = [ "work experience", "employment history", "professional experience", "education", "skills", "certifications", "references", "volunteer", "objective", "summary", "professional summary", "career objective", "technical skills", "work history", "job history", "activities", "achievements", "accomplishments", "projects", "portfolio", "languages", "proficient in", "expertise in", "competencies" ]
            indicator_count = sum(1 for indicator in resume_indicators if indicator in text_lower)
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
            phone_pattern = r'\b(?:(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{2,4}\b|\d{10})\b'
            linkedin_pattern = r"linkedin\.com/in/[\w-]+"
            has_email = bool(re.search(email_pattern, text)); has_phone = bool(re.search(phone_pattern, text)); has_linkedin = bool(re.search(linkedin_pattern, text_lower))
            personal_info_count = sum([has_email, has_phone, has_linkedin])
            weighted_score = (indicator_count * 2) + personal_info_count
            expected_score_threshold = 10 # Adjust based on testing
            confidence = min(1.0, weighted_score / expected_score_threshold)
            is_valid_threshold_confidence = 0.4; is_valid_min_indicators = 3; is_valid_min_indicators_with_contact = 2
            is_valid = ( confidence >= is_valid_threshold_confidence and (indicator_count >= is_valid_min_indicators or (indicator_count >= is_valid_min_indicators_with_contact and personal_info_count > 0)))
            indicators_found = indicator_count; personal_info_found = personal_info_count
        except Exception as e:
             print(f"Error during local resume validation logic: {e}"); traceback.print_exc(); error_message = f"Internal error during text analysis: {e}"; is_valid = False; indicators_found = 0; personal_info_found = 0; confidence = 0.0
        final_results = { "is_resume": is_valid, "confidence": round(confidence, 3), "indicators_found": indicators_found, "personal_info_found": personal_info_found, "top_label": "resume" if is_valid else "other", "service_used": "local_keyword_analysis" }
        if error_message: final_results["error"] = error_message
        return is_valid, final_results

    def is_resume(self, text):
        """Primary method to validate resume text."""
        print("Using local keyword analysis for resume classification.")
        try: return self.is_resume_local(text)
        except Exception as e:
            print(f"Unhandled error in is_resume_local: {e}"); traceback.print_exc()
            return False, { "is_resume": False, "confidence": 0.0, "error": f"Validation analysis failed: {e}", "service_used": "local_keyword_analysis" }

class ValidateResumeView(APIView):
    permission_classes = [AllowAny] # Or IsAuthenticated

    def post(self, request, format=None):
        file = request.FILES.get('file')
        if not file: return Response({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)
        if not file.name.lower().endswith('.pdf'): return Response({"error": "File must be a PDF"}, status=status.HTTP_400_BAD_REQUEST)

        validator = ResumeValidator()
        try:
            file.seek(0); text = validator.extract_text(file); is_valid, results = validator.is_resume(text)
            # Return results directly, they should be serializable now
            return Response({ 'is_resume': is_valid, 'confidence': results.get('confidence', 0), 'top_label': results.get('top_label', 'unknown'), 'service_used': results.get('service_used', 'unknown'), 'details': results }, status=status.HTTP_200_OK)
        except (PyPDF2Errors.PdfReadError, ValueError, ImportError) as e:
            print(f"ERROR validating document '{getattr(file,'name','N/A')}': {e}"); status_code = status.HTTP_500_INTERNAL_SERVER_ERROR; error_detail = str(e)
            if isinstance(e, PyPDF2Errors.PdfReadError): status_code=status.HTTP_400_BAD_REQUEST; error_detail = f"Could not read PDF: {e}"
            elif isinstance(e, ValueError): status_code=status.HTTP_400_BAD_REQUEST; error_detail = f"Processing error: {e}"
            elif isinstance(e, ImportError): error_detail = "Server configuration error (missing library)."
            return Response({"error": "Error validating document", "details": error_detail}, status=status_code)
        except Exception as e:
            print(f"Unexpected error during validation for '{getattr(file,'name','N/A')}': {e}"); traceback.print_exc()
            return Response({"error": "An unexpected server error occurred.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class UploadResumeView(APIView):
    permission_classes = [IsAuthenticated] # Require login

    def post(self, request, format=None):
        file = request.FILES.get('file')
        validate_only_str = str(request.data.get('validate_only', 'false')).lower()
        validate_only = validate_only_str == 'true'

        if not file: return Response({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)
        if not file.name.lower().endswith('.pdf'): return Response({"error": "File must be a PDF"}, status=status.HTTP_400_BAD_REQUEST)

        validator = ResumeValidator()
        try:
            file.seek(0); extracted_text = validator.extract_text(file); file.seek(0)
            is_valid, results = validator.is_resume(extracted_text)
            print(f"DEBUG: Validation result: is_valid={is_valid}, service={results.get('service_used')}")

            if validate_only:
                 return Response({ 'is_resume': is_valid, 'confidence': results.get('confidence', 0), 'top_label': results.get('top_label', 'unknown'), 'service_used': results.get('service_used', 'unknown'), 'details': results }, status=status.HTTP_200_OK)

            if not is_valid:
                 return Response({ "error": "The uploaded file doesn't appear to be a resume.", "details": results }, status=status.HTTP_400_BAD_REQUEST)

            authenticated_user = request.user
            print(f"DEBUG: Validation passed. Saving resume for user: {authenticated_user.username}")
            current_role = get_user_role(authenticated_user)

            # Use imported USER_ROLE_CHOICES
            if current_role not in dict(USER_ROLE_CHOICES):
                 print(f"WARNING: User {authenticated_user.username} has unrecognized role '{current_role}'. Defaulting resume role to 'jobseeker'.")
                 current_role_for_resume = 'jobseeker'
            else:
                 current_role_for_resume = current_role

            print(f"DEBUG: Saving resume with role: {current_role_for_resume}")
            resume = Resume(user=authenticated_user, uploaded_as_role=current_role_for_resume, file=file, extracted_text=extracted_text, title=os.path.splitext(file.name)[0])
            resume.save()
            print(f"DEBUG: Resume saved: ID {resume.id}, Role {resume.uploaded_as_role}")

            serializer = ResumeSerializer(resume)
            return Response(serializer.data, status=status.HTTP_201_CREATED)

        except (PyPDF2Errors.PdfReadError, ValueError, ImportError, IntegrityError, Exception) as e:
            file_name = getattr(file, 'name', 'N/A') # Get filename safely
            print(f"ERROR during resume upload for file '{file_name}': {e}")
            traceback.print_exc(); error_message = "An unexpected server error occurred."; status_code = status.HTTP_500_INTERNAL_SERVER_ERROR; details = "Please try again later."
            if isinstance(e, PyPDF2Errors.PdfReadError): error_message = "PDF Read Error"; details = f"File '{file_name}' unreadable: {e}"; status_code = status.HTTP_400_BAD_REQUEST
            elif isinstance(e, ValueError): error_message = "Processing Error"; details = str(e); status_code = status.HTTP_400_BAD_REQUEST
            elif isinstance(e, ImportError): error_message = "Server Config Error"; details = "Required library missing."
            elif isinstance(e, IntegrityError): error_message = "Database Conflict"; details = "Failed to save resume."; status_code = status.HTTP_400_BAD_REQUEST
            return Response({"error": error_message, "details": details}, status=status_code)

