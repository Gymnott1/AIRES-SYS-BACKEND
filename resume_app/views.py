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

# For authentication views:
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from rest_framework.authtoken.models import Token
from django.contrib.auth import update_session_auth_hash





# Get your Hugging Face API key from an environment variable
HF_API_KEY = "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"


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
        if not file:
            return Response({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)
        
        file.seek(0)
        resume = Resume(file=file)
        # If user is authenticated, associate the resume with them.
        if request.user.is_authenticated:
            resume.user = request.user
        resume.save()

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
        scores (skills, experience, education, overall) as percentages, and. Also, provide exactly 10 key insights and exactly 10 actionable improvement suggestions. The key insights and improvement suggestions must cover the following areas:
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
            "You are an expert resume advisor. Your answer must reference specific details from the CV provided below. "
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




