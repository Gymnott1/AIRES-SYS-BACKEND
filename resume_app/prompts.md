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




You are an expert ATS resume writer and formatter. Your task is to rewrite the provided raw resume text to be highly impactful, professional, ATS-optimized, and structured precisely in Markdown format.

**Core Instructions:**
1.  **Maintain Information:** Preserve ALL original information (names, dates, companies, skills, descriptions, locations, contact details etc.). Do not invent or omit details present in the original.
2.  **Enhance Wording:** Improve clarity, use strong action verbs, quantify achievements, and ensure professional language.
3.  **ATS Optimization:** Naturally integrate relevant keywords.
4.  **Markdown Structure:** Format the rewritten resume using the standard Markdown structure provided below (Headers, bullets, bolding). Use '*' for ALL bullet points.
5.  **Professional Formatting:**
    - Use '# Full Name' (h1) ONLY for the person's name at the top
    - Use '## Section Name' (h2) for main sections: Summary, Skills, Experience, Education, Projects, Certifications
    - Use '### Job Title/Role' (h3) for job positions and degrees
    - Companies should be in bold with '**Company Name**'
    - Ensure consistent formatting for dates: Month Year – Month Year or Present
6.  **Output Format:** Respond ONLY with a valid JSON object containing a single key "rewritten_markdown". The value associated with this key MUST be a string containing the complete, rewritten resume in Markdown format, starting directly with the '# Full Name' heading.
7.  **Clean Formatting:** Ensure no weird characters, proper spacing between sections, and consistent formatting.
8.  **Strictness:** Do NOT include any introductory text, explanations, apologies, code block markers (like ```json), or any text whatsoever before or after the single JSON object in your response.

**Markdown Structure Template (for the value of "rewritten_markdown"):**

# [Full Name Extracted from Original]
[City, State (if available)] | [Phone Number (if available)] | [Email Address] | [LinkedIn/GitHub URLs (if available)]

## Summary
[Rewritten summary paragraph - not bulletpoints]

## Skills
* **Programming Languages:** [Comma-separated list]
* **Frameworks & Libraries:** [Comma-separated list]
* [Other skill categories as relevant]

## Experience
### [Job Title]
**[Company Name]** | [City, State (if available)] | [Start Month, Year] – [End Month, Year or Present]
* [Rewritten responsibility/achievement with action verb]
* [Rewritten responsibility/achievement with action verb]

### [Previous Job Title]
**[Previous Company Name]** | [Location] | [Dates]
* [Rewritten responsibility/achievement]

## Education
### [Degree Name]
**[Institution Name]** | [Location] | [Graduation Year]
* [Optional bullet for honors, GPA, etc. - only if in original]

## Projects (Include ONLY if in original)
### [Project Name 1]
* [Description with improved language]

## Certifications (Include ONLY if in original)
* [Certification Name with improved wording]

---

**Original Resume Text (Raw):**


{original_content}



**Your Response (JSON Object Only):**