Analyze the following resume and return ONLY valid JSON (with no additional text or formatting) that exactly follows the structure below. Evaluate the resume and assign percentage scores (0–100) for each area 
scores (skills, experience, education, overall) as percentages, and. Also, provide exactly 10 key insights and exactly 10 actionable improvement suggestions. The key insights and improvement suggestions must cover the following areas:
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

The expected JSON structure is:

(scores (skills, experience, education, overall) , key_insights (insight 1, insight 2, ... (exactly 10 insights) improvement_suggestions( suggestion 1,  suggestion 2,  ... (exactly 10 suggestions) )  ))
   
  
Resume: {resume.text}


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
