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
# For authentication views:
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from rest_framework.authtoken.models import Token
from django.contrib.auth import update_session_auth_hash


from django.http import HttpResponse
from reportlab.lib.pagesizes import letter
import re

HF_API_KEY = "PUT-YOUR-HUGGINGFACE-API-KEY"


def clean_json_string(json_str):
    # Replace backslash-newline sequences with a newline.
    cleaned = json_str.replace('\\\n', '\n')
    # Remove control characters that might invalidate the JSON.
    cleaned = re.sub(r'[\x00-\x1F\x7F]', '', cleaned)
    return cleaned


class ResumeValidator:
    def __init__(self):
        self.api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
        self.headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    
    def extract_text(self, pdf_file):
        # Extract text from PDF using PyPDF2
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text  # Add this return statement
    def is_resume(self, text):
        # Prepare payload for zero-shot classification
        payload = {
            "inputs": text[:1024],  # Limit text length to avoid token limits
            "parameters": {
                "candidate_labels": [
                    "resume", "curriculum vitae", "CV", "job application",
                    "article", "report", "manual", "academic paper", "letter",
                    "other"
                ]
            }
        }
        
        # Call the Hugging Face API
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        result = response.json()
        print (result)
        
        
        # Check if resume-related labels are ranked highest
        resume_labels = ["resume", "curriculum vitae", "CV", "job application", "academic paper"]
        
        if "labels" in result and "scores" in result:
            top_label = result["labels"][0] 
            top_score = result["scores"][0]
            
            is_valid = top_label in resume_labels
            
            return is_valid, {
                "is_resume": is_valid,
                "confidence": top_score,
                "top_label": top_label,
                "details": result
            }
        
        return False, {"error": "Classification failed", "details": result}


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
    

class AnalyzeResumeView(APIView):
    permission_classes = [AllowAny]
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
        
        # Create a prompt for the Hugging Face model
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
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {"inputs": prompt, "parameters": {"max_tokens": 10000}}

        try:
            hf_response = requests.post(
                "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
                headers=headers,
                json=payload
            )
            result = hf_response.json()
            print("Hugging Face API response:", result)  # Check what the API returns

            if "error" in result:
                return Response(
                    {"error": "Error analyzing resume", "details": result["error"]},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            if not isinstance(result, list) or len(result) == 0:
                return Response(
                    {"error": "Unexpected API response format", "details": result},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

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
        except Exception as e:
            return Response(
                {"error": "Error analyzing resume", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        


class ChatView(APIView):
    permission_classes = [AllowAny]  # Allow any user to chat
    def post(self, request, format=None):
        resume_id = request.data.get("resume_id")
        message = request.data.get("message")
        conversation = request.data.get("conversation", [])
        
        if not resume_id or not message:
            return Response({"error": "Missing resume_id or message"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Retrieve the resume object from the database
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
        
        # Debug: Print the resume text for verification
        print("DEBUG: Resume text content:")
        print(resume_text)
        
        # Construct the prompt for the AI
        chat_prompt = (
            "You are an expert ATS resume advisor. Your answer must reference specific details from the CV provided below. "
            "Do not provide generic advice. Instead, analyze the CV content (including skills, education, experience, achievements, etc.) "
            "and tailor your answer based on that information. If the CV lacks sufficient details, mention it explicitly. Do not exceed 100 words.\n\n"
            "CV Content:\n" + resume_text + "\n\n" +
            "Based on the CV above, please answer the following question, referencing specific details from the CV:\n" +
            "User: " + message + "\n" +
            "AI:"
        )
        
        payload = {
            "inputs": chat_prompt,
            "parameters": {"max_tokens": 500}
        }
        headers = {
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json"
        }
        
        try:
            ai_response = requests.post(
                "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
                headers=headers,
                json=payload
            )
            print("DEBUG: Hugging Face API raw response:", ai_response.text)
            ai_result = ai_response.json()
            if isinstance(ai_result, list) and len(ai_result) > 0:
                reply = ai_result[0].get("generated_text", "Sorry, no response from AI.")
            else:
                reply = "Sorry, no response from AI."
            
            # Split the reply on "AI:" and return only the final segment
            parts = reply.split("AI:")
            final_reply = parts[-1].strip() if parts else reply
            
            # If the user is authenticated, save the AI response in the database
            if request.user.is_authenticated:
                ChatMessage.objects.create(
                    resume=resume_obj,
                    user=request.user,
                    sender='ai',
                    message=final_reply
                )
            
            return Response({"reply": final_reply}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": "Chat processing failed", "details": str(e)},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)



# Authentication Views
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




@api_view(['POST'])
@permission_classes([AllowAny])
def rewrite_resume(request):
    """
    Endpoint to rewrite a resume using AI, requesting and parsing JSON output,
    including cleaning of invalid escape sequences.
    """
    resume_id = request.data.get('resume_id')
    if not resume_id:
        return Response({'error': 'Resume ID not provided'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        # --- Get the original resume content ---
        resume = Resume.objects.get(id=resume_id)

        # Check if text is available, if not extract it
        if not resume.text:
            try:
                # Ensure the file path is correct and accessible
                if not resume.file or not hasattr(resume.file, 'path') or not os.path.exists(resume.file.path):
                     raise FileNotFoundError(f"Resume file path not found or invalid: {getattr(resume.file, 'path', 'N/A')}")

                with open(resume.file.path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    extracted_text = ""
                    if not pdf_reader.pages:
                         # Handle case where PDF has no pages or is encrypted badly
                         print(f"Warning: PDF for resume {resume_id} has no pages or could not be read.")
                         raise ValueError("PDF contains no pages or is unreadable.")
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                             extracted_text += page_text + "\n" # Add newline between pages

                if not extracted_text.strip():
                    # Handle case where text extraction yields nothing
                    print(f"Warning: Could not extract text from PDF for resume {resume_id}.")
                    raise ValueError("Could not extract text from PDF.")

                resume.text = extracted_text
                resume.save(update_fields=['text']) # Save only the text field
                print(f"DEBUG: Extracted text from PDF for resume {resume_id}")
            except (FileNotFoundError, ValueError, PyPDF2.errors.PdfReadError, Exception) as extraction_error:
                 print(f"Error extracting text from PDF for resume {resume_id}: {extraction_error}")
                 # Provide a more specific error message if possible
                 error_detail = str(extraction_error)
                 if isinstance(extraction_error, PyPDF2.errors.PdfReadError):
                     error_detail = "Could not read the PDF file, it might be corrupted or encrypted."

                 return Response(
                     {'error': 'Failed to process the original resume PDF.', 'details': error_detail},
                     status=status.HTTP_500_INTERNAL_SERVER_ERROR
                 )

        original_content = resume.text
        if not original_content or not original_content.strip():
             # This case should be less likely after the extraction improvements, but keep as safety
             print(f"ERROR: Original resume content is empty for resume {resume_id} even after extraction attempt.")
             return Response({'error': 'Original resume content is empty.'}, status=status.HTTP_400_BAD_REQUEST)

        # --- Create prompt for Hugging Face API (Requesting JSON) ---
        # (Keep the same detailed JSON prompt as provided in the previous answer)
        prompt = f"""
You are an expert ATS resume writer and formatter. Your task is to rewrite the provided raw resume text to be highly impactful, professional, ATS-optimized, and structured precisely in Markdown format.

**Core Instructions:**
1.  **Maintain Information:** Preserve ALL original information (names, dates, companies, skills, descriptions, locations, contact details etc.). Do not invent or omit details present in the original.
2.  **Enhance Wording:** Improve clarity, use strong action verbs, quantify achievements, and ensure professional language.
3.  **ATS Optimization:** Naturally integrate relevant keywords.
4.  **Markdown Structure:** Format the rewritten resume using the standard Markdown structure provided below (Headers, bullets, bolding). Use '*' for ALL bullet points.
5.  **Output Format:** Respond ONLY with a valid JSON object containing a single key "rewritten_markdown". The value associated with this key MUST be a string containing the complete, rewritten resume in Markdown format, starting directly with the '# Full Name' heading.
6.  **Strictness:** Do NOT include any introductory text, explanations, apologies, code block markers (like ```json), or any text whatsoever before or after the single JSON object in your response.

**Markdown Structure Template (for the value of "rewritten_markdown"):**

# [Full Name Extracted from Original]
[City, State (if available)] | [Phone Number (if available)] | [Email Address] | [LinkedIn Profile URL (if available, otherwise omit)]

## Summary
[Rewritten summary text...]

## Skills
*   **Programming Languages:** [Comma-separated list]
*   **Frameworks & Libraries:** [Comma-separated list]
*   [...]

## Experience
### [Job Title]
**[Company Name]** | [City, State] | [Start Month, Year] – [End Month, Year or Present]
*   [Rewritten responsibility/achievement 1...]
*   [Rewritten responsibility/achievement 2...]

### [Previous Job Title]
**[Previous Company Name]** | [...]
*   [...]

## Education
### [Degree Name]
**[Institution Name]** | [...]
*   [Optional bullet...]

## Projects (Include ONLY if distinct)
### [Project Name 1]
*   [Description...]

## Certifications (Include ONLY if mentioned)
*   [Certification Name...]

---

**Original Resume Text (Raw):**


{original_content}



**Your Response (JSON Object Only):**
"""

        headers = {
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "inputs": prompt,
            "parameters": {
                 "max_new_tokens": 2500,
                 "return_full_text": False,
                 "temperature": 0.7,
                 "do_sample": True,
                 # Consider adding repetition penalty if needed: "repetition_penalty": 1.1
                 }
        }

        rewritten_content = None # Initialize variable

        try:
            # --- Try sending request to Hugging Face API ---
            print(f"DEBUG: Sending request to Hugging Face API for resume {resume_id}")
            response = requests.post(
                "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2", # Or your chosen model
                headers=headers,
                json=payload,
                timeout=45
            )
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            result = response.json()
            raw_generated_text = "" # Initialize

            # Handle potential variations in successful response structure
            if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
                 raw_generated_text = result[0].get("generated_text", "")
            elif isinstance(result, dict) and "generated_text" in result:
                 raw_generated_text = result.get("generated_text", "")
            else:
                 print(f"Unexpected API response format: {result}")
                 raise Exception("Unexpected response format from AI service")

            print(f"DEBUG: Raw AI Response Text for resume {resume_id}:\n---\n{raw_generated_text}\n---")

            # --- Attempt to parse JSON output ---
            try:
                json_start = raw_generated_text.find('{')
                json_end = raw_generated_text.rfind('}')
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    json_str = raw_generated_text[json_start: json_end + 1]
                    print(f"DEBUG: Extracted Raw JSON String:\n---\n{json_str}\n---")

                    # --- Use the cleaning function ---
                    cleaned_json_str = clean_json_string(json_str)
                    print(f"DEBUG: Cleaned JSON String for Parsing:\n---\n{cleaned_json_str}\n---")

                    try:
                        parsed_json = json.loads(cleaned_json_str)
                        if "rewritten_markdown" in parsed_json and isinstance(parsed_json["rewritten_markdown"], str):
                            rewritten_content = parsed_json["rewritten_markdown"].strip()
                            # Further clean common AI artifacts if necessary
                            rewritten_content = rewritten_content.replace("```json\n", "").replace("\n```", "")
                            print("DEBUG: Successfully parsed CLEANED JSON and extracted rewritten_markdown.")
                        else:
                            print("DEBUG: Cleaned JSON parsed, but 'rewritten_markdown' key missing or not a string.")
                            raise ValueError("Invalid JSON structure received from AI (after cleaning).")
                    except json.JSONDecodeError as e:
                        print(f"DEBUG: JSON decoding error after cleaning: {e}")
                        # Optionally, you can add another cleaning attempt here
                else:
                    print("DEBUG: JSON markers ({...}) not found in the response.")
                    raise ValueError("Could not find JSON object markers in AI response.")


            except (json.JSONDecodeError, ValueError) as parse_error:
                print(f"Warning: Failed to parse JSON from AI response ({parse_error}). Attempting fallback.")
                # --- Fallback: Try Marker-Based Extraction ---
                # Use a marker expected *after* the main instructions
                marker = "**Your Response (JSON Object Only):**"

                if marker in raw_generated_text:
                    # Get content *after* the final instruction marker
                    parts = raw_generated_text.rsplit(marker, 1)
                    if len(parts) > 1:
                         potential_content = parts[1].strip()
                         # Basic check if it looks like markdown start
                         if potential_content.startswith("#"):
                              rewritten_content = potential_content
                              # Clean potential stray ``` markers from fallback too
                              rewritten_content = rewritten_content.replace("```json\n", "").replace("\n```", "").replace("```", "")
                              print("DEBUG: Fallback marker extraction successful.")
                         else:
                              print(f"DEBUG: Fallback marker found, but content doesn't look like expected markdown. Starts with: {potential_content[:50]}...")
                    else:
                         print("DEBUG: Fallback marker found, but split failed.")
                else:
                     print("DEBUG: Fallback marker also not found.")
                # If fallback also failed, rewritten_content will still be None here


        except requests.exceptions.RequestException as api_error:
            print(f"Error calling Hugging Face API: {api_error}")
            # Log more details if possible (e.g., response content for 4xx/5xx errors)
            if hasattr(api_error, 'response') and api_error.response is not None:
                print(f"API Error Response Status: {api_error.response.status_code}")
                print(f"API Error Response Body: {api_error.response.text[:500]}...") # Log first 500 chars
            # Keep rewritten_content as None to trigger mock data below

        except Exception as e:
             # Catch other unexpected errors during API call/processing
             print(f"Unexpected error during AI processing: {e}")
             import traceback
             traceback.print_exc() # Log full traceback for server logs
             # Keep rewritten_content as None


        # --- Final Check and Mock Data Fallback ---
        if rewritten_content is None or not rewritten_content.strip(): # Check if it's None or empty/whitespace
            print("ERROR: Failed to get valid content from AI after JSON and fallback attempts. Using mock data.")
            # Your existing mock data fallback
            rewritten_content = f"""
# JOHN DOE
New York, NY | (555) 123-4567 | johndoe@email.com | linkedin.com/in/johndoe

## Summary
Results-driven software engineer with 5+ years of experience building scalable web applications. Expertise in React, Node.js, and cloud architecture. Strong problem-solving skills with a focus on delivering high-quality code and excellent user experiences.

## Skills
*   **Programming Languages:** JavaScript, Python, TypeScript, SQL
*   **Frameworks & Libraries:** React, Node.js, Express, Django, Redux, TailwindCSS
*   **Tools & Platforms:** Git, Docker, AWS, CI/CD, Jira, Agile methodologies

## Experience
### SENIOR SOFTWARE ENGINEER
**ABC Tech** | New York, NY | January 2020 – Present
*   Led development of new customer portal that improved user engagement by 35%.
*   Architected microservice infrastructure that reduced deployment time by 40%.
*   Mentored 5 junior developers, conducting code reviews and technical training.

### SOFTWARE ENGINEER
**XYZ Solutions** | Boston, MA | June 2017 – December 2019
*   Developed RESTful APIs for integration with partner platforms, increasing revenue by 20%.
*   Optimized database queries, reducing page load times by 60%.
*   Implemented automated testing suite that increased code coverage from 65% to 92%.

## Education
### MASTER OF SCIENCE IN COMPUTER SCIENCE
**Massachusetts Institute of Technology** | Cambridge, MA | 2017

### BACHELOR OF SCIENCE IN COMPUTER ENGINEERING
**University of California, Berkeley** | Berkeley, CA | 2015

[Note: This is mock data as the AI service response could not be processed reliably.]
"""
            message = "Resume rewrite failed to process AI response, using placeholder data."
        else:
            message = "Resume rewrite processed successfully using AI response."


        # --- Store the rewritten content in the database ---
        try:
            resume.rewritten_content = rewritten_content
            resume.save(update_fields=['rewritten_content'])
            print(f"DEBUG: Saved rewritten content (length: {len(rewritten_content)}) for resume {resume_id}")
        except Exception as db_error:
             print(f"ERROR: Failed to save rewritten content to database for resume {resume_id}: {db_error}")
             # Decide how to handle this - return error? Return content without saving?
             return Response(
                {'error': 'Failed to save the rewritten resume content.', 'details': str(db_error)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
             )


        return Response({
            'rewritten_content': rewritten_content,
            'message': message # Updated message reflects outcome
        }, status=status.HTTP_200_OK)

    except Resume.DoesNotExist:
        return Response(
            {'error': 'Resume not found'},
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        # Catch-all for unexpected errors (e.g., database connection issues)
        print(f"Fatal error in rewrite_resume view for resume {resume_id}: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for server logs
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
        
        # Create prompt for Hugging Face API (Requesting JSON)
        prompt = f"""
You are an expert ATS resume writer and formatter. Your task is to revise the provided resume based on the user's feedback, while maintaining professional ATS formatting and style.

**Core Instructions:**
1.  **Make Requested Changes:** Apply the user's feedback carefully, preserving the overall professional quality.
2.  **Maintain Information:** Preserve ALL original information that the user doesn't ask to change.
3.  **Enhance Wording:** Improve clarity, use strong action verbs, quantify achievements, and ensure professional language.
4.  **ATS Optimization:** Naturally integrate relevant keywords.
5.  **Markdown Structure:** Format the revised resume using the standard Markdown structure provided below.
6.  **Output Format:** Respond ONLY with a valid JSON object containing a single key "revised_markdown". The value associated with this key MUST be a string containing the complete, revised resume in Markdown format, starting directly with the '# Full Name' heading.
7.  **Strictness:** Do NOT include any introductory text, explanations, apologies, code block markers (like ```json), or any text whatsoever before or after the single JSON object in your response.

**Markdown Structure Template (for the value of "revised_markdown"):**

# [Full Name]
[City, State (if available)] | [Phone Number (if available)] | [Email Address] | [LinkedIn Profile URL (if available, otherwise omit)]

## Summary
[Revised summary text...]

## Skills
*   **Programming Languages:** [Comma-separated list]
*   **Frameworks & Libraries:** [Comma-separated list]
*   [...]

## Experience
### [Job Title]
**[Company Name]** | [City, State] | [Start Month, Year] – [End Month, Year or Present]
*   [Revised responsibility/achievement 1...]
*   [Revised responsibility/achievement 2...]

### [Previous Job Title]
**[Previous Company Name]** | [...]
*   [...]

## Education
### [Degree Name]
**[Institution Name]** | [...]
*   [Optional bullet...]

## Projects (Include ONLY if distinct)
### [Project Name 1]
*   [Description...]

## Certifications (Include ONLY if mentioned)
*   [Certification Name...]

---

**Current Resume:**
{current_version}

**User Feedback:**
{feedback}

**Your Response (JSON Object Only):**
"""
        
        headers = {
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                 "max_new_tokens": 2500,
                 "return_full_text": False,
                 "temperature": 0.7,
                 "do_sample": True,
                 # Consider adding repetition penalty if needed: "repetition_penalty": 1.1
            }
        }
        
        revised_content = None  # Initialize variable
        
        try:
            # --- Try sending request to Hugging Face API ---
            print(f"DEBUG: Sending request to Hugging Face API for resume revision {resume_id}")
            response = requests.post(
                "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
                headers=headers,
                json=payload,
                timeout=45
            )
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            
            result = response.json()
            raw_generated_text = ""  # Initialize
            
            # Handle potential variations in successful response structure
            if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
                raw_generated_text = result[0].get("generated_text", "")
            elif isinstance(result, dict) and "generated_text" in result:
                raw_generated_text = result.get("generated_text", "")
            else:
                print(f"Unexpected API response format: {result}")
                raise Exception("Unexpected response format from AI service")
            
            print(f"DEBUG: Raw AI Response Text for resume revision {resume_id}:\n---\n{raw_generated_text}\n---")
            
            # --- Attempt to parse JSON output ---
            try:
                json_start = raw_generated_text.find('{')
                json_end = raw_generated_text.rfind('}')
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    json_str = raw_generated_text[json_start: json_end + 1]
                    print(f"DEBUG: Extracted Raw JSON String:\n---\n{json_str}\n---")
                    
                    # --- Use the cleaning function ---
                    cleaned_json_str = clean_json_string(json_str)
                    print(f"DEBUG: Cleaned JSON String for Parsing:\n---\n{cleaned_json_str}\n---")
                    
                    try:
                        parsed_json = json.loads(cleaned_json_str)
                        if "revised_markdown" in parsed_json and isinstance(parsed_json["revised_markdown"], str):
                            revised_content = parsed_json["revised_markdown"].strip()
                            # Further clean common AI artifacts if necessary
                            revised_content = revised_content.replace("```json\n", "").replace("\n```", "")
                            print("DEBUG: Successfully parsed CLEANED JSON and extracted revised_markdown.")
                        else:
                            print("DEBUG: Cleaned JSON parsed, but 'revised_markdown' key missing or not a string.")
                            raise ValueError("Invalid JSON structure received from AI (after cleaning).")
                    except json.JSONDecodeError as e:
                        print(f"DEBUG: JSON decoding error after cleaning: {e}")
                        # Optionally, you can add another cleaning attempt here
                else:
                    print("DEBUG: JSON markers ({...}) not found in the response.")
                    raise ValueError("Could not find JSON object markers in AI response.")
                
            except (json.JSONDecodeError, ValueError) as parse_error:
                print(f"Warning: Failed to parse JSON from AI response ({parse_error}). Attempting fallback.")
                # --- Fallback: Try Marker-Based Extraction ---
                # Use a marker expected *after* the main instructions
                marker = "**Your Response (JSON Object Only):**"
                
                if marker in raw_generated_text:
                    # Get content *after* the final instruction marker
                    parts = raw_generated_text.rsplit(marker, 1)
                    if len(parts) > 1:
                        potential_content = parts[1].strip()
                        # Basic check if it looks like markdown start
                        if potential_content.startswith("#"):
                            revised_content = potential_content
                            # Clean potential stray ``` markers from fallback too
                            revised_content = revised_content.replace("```json\n", "").replace("\n```", "").replace("```", "")
                            print("DEBUG: Fallback marker extraction successful.")
                        else:
                            print(f"DEBUG: Fallback marker found, but content doesn't look like expected markdown. Starts with: {potential_content[:50]}...")
                    else:
                        print("DEBUG: Fallback marker found, but split failed.")
                else:
                    print("DEBUG: Fallback marker also not found.")
                    
        except requests.exceptions.RequestException as api_error:
            print(f"Error calling Hugging Face API: {api_error}")
            # Log more details if possible (e.g., response content for 4xx/5xx errors)
            if hasattr(api_error, 'response') and api_error.response is not None:
                print(f"API Error Response Status: {api_error.response.status_code}")
                print(f"API Error Response Body: {api_error.response.text[:500]}...")  # Log first 500 chars
            # Keep revised_content as None to trigger mock data below
            
        except Exception as e:
            # Catch other unexpected errors during API call/processing
            print(f"Unexpected error during AI processing: {e}")
            import traceback
            traceback.print_exc()  # Log full traceback for server logs
            # Keep revised_content as None
            
        # --- Final Check and Mock Data Fallback ---
        if revised_content is None or not revised_content.strip():  # Check if it's None or empty/whitespace
            print("ERROR: Failed to get valid content from AI after JSON and fallback attempts. Using existing content with feedback note.")
            # Add the feedback as a note to the existing content
            revised_content = current_version + f"\n\n# REVISION BASED ON FEEDBACK: \n{feedback}\n\n[Note: This is a placeholder as the AI service response could not be processed reliably.]"
            message = "Resume revision failed to process AI response, using placeholder data with feedback note."
        else:
            message = "Resume revision processed successfully using AI response."
            
        # --- Additional cleaning to ensure proper formatting for PDF generation ---
        # Remove any triple backticks that might interfere with markdown parsing
        revised_content = revised_content.replace("```markdown", "").replace("```", "")
        
        # Ensure proper line breaks for markdown
        revised_content = re.sub(r'\n{3,}', '\n\n', revised_content)  # Replace excessive newlines
        
        # Make sure all bullets are properly formatted for markdown-to-html conversion
        revised_content = re.sub(r'(?<=\n)\s*\*\s+', '* ', revised_content)
        
        # --- Store the revised content in the database ---
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
            'message': message  # Updated message reflects outcome
        }, status=status.HTTP_200_OK)
        
    except Resume.DoesNotExist:
        return Response(
            {'error': 'Resume not found'},
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        # Catch-all for unexpected errors (e.g., database connection issues)
        print(f"Fatal error in revise_resume view for resume {resume_id}: {e}")
        import traceback
        traceback.print_exc()  # Print full traceback for server logs
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
        
        # Remove any lingering code block markers
        content_md = content_md.replace("```markdown", "").replace("```json", "").replace("```", "")
        
        # Ensure proper line breaks
        content_md = re.sub(r'\n{3,}', '\n\n', content_md)  # Replace excessive newlines
        
        # Ensure bullet points are properly formatted
        content_md = re.sub(r'(?<=\n)\s*\*\s+', '* ', content_md)
        
        # Ensure headings have proper spacing
        content_md = re.sub(r'(?<=\n)#{1,6}\s*', lambda m: m.group().strip() + ' ', content_md)
        
        # 1. Convert Markdown to HTML
        print("DEBUG: Starting Markdown to HTML conversion.")
        html_content = markdown(content_md, extensions=['fenced_code', 'tables', 'nl2br'])
        print("DEBUG: Markdown to HTML conversion finished.")

        # 2. Add CSS classes to enhance structure for PDF generation
        print("DEBUG: Adding structure for PDF formatting")
        # Add classes to job entries for better page breaks
        html_content = re.sub(r'<h3>(.*?)</h3>', r'<h3 class="job-title">\1</h3>', html_content)
        
        # Wrap each job section in a div with a class
        sections = re.findall(r'<h3 class="job-title">.*?(?=<h3 class="job-title">|<h2|$)', html_content, re.DOTALL)
        for section in sections:
            html_content = html_content.replace(section, f'<div class="job-entry">{section}</div>')
        
        # Same for education sections
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
            # Save in a predictable location, maybe MEDIA_ROOT or a specific debug folder
            # Ensure the target directory exists and has write permissions
            debug_dir = os.path.join(settings.BASE_DIR, 'debug_output') # Or settings.MEDIA_ROOT
            os.makedirs(debug_dir, exist_ok=True)
            debug_html_path = os.path.join(debug_dir, "debug_resume_output.html")
            with open(debug_html_path, "w", encoding="utf-8") as f:
                f.write(full_html)
            print(f"DEBUG: Saved intermediate HTML to {debug_html_path}")
        except Exception as html_save_error:
            print(f"Warning: Could not save debug HTML file: {html_save_error}")
        # --- END OF SAVE HTML ---

        # 3. Create PDF using pisa
        print("DEBUG: Starting PDF generation with pisa.")
        pdf_buffer = io.BytesIO()
        pisa_status = pisa.CreatePDF(
            src=io.BytesIO(full_html.encode('UTF-8')),
            dest=pdf_buffer,
            encoding='UTF-8'
            # link_callback=link_callback # Uncomment if using local images/resources
        )
        print(f"DEBUG: pisa.CreatePDF finished. Success: {not pisa_status.err}")

        # 4. Check for errors during PDF generation
        if pisa_status.err:
            print(f"xhtml2pdf Error Code: {pisa_status.err}") # Log the error code
            # Attempt to get more detailed logging from pisa if available
            error_details = f"PDF Generation Error Code: {pisa_status.err}"
            detailed_log = pisa_status.log # Accessing the log attribute
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

            # Also include the first part of the HTML that failed
            error_details += f"\n\n--- Failing HTML (approx first 500 chars) ---\n{full_html[:500]}"

            return Response({'error': 'PDF generation failed', 'details': error_details}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # 5. Prepare and Return Successful Response
        print("DEBUG: PDF generated successfully. Preparing response.")
        pdf_buffer.seek(0)
        response = HttpResponse(pdf_buffer, content_type='application/pdf')
        response['Content-Disposition'] = 'attachment; filename="improved_resume.pdf"'
        return response

    except Exception as e:
        # Catch-all for other errors
        print(f"Error in generate_pdf view: {e}")
        import traceback
        traceback.print_exc()
        return Response({'error': 'An internal server error occurred during PDF generation.', 'details': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)