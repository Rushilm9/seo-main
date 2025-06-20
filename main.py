import os
import streamlit as st
import requests
import pandas as pd
from dotenv import load_dotenv
from io import BytesIO

# Load environment variables
load_dotenv()

# --- Azure OpenAI LLM Setup ---
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0
)

# --- SEMrush API Key ---
SEMRUSH_API_KEY = os.getenv("SEMRUSH_API_KEY")

# --- Tavily API Key ---
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# --- Helper Functions ---

def tavily_search(query, num_results=10):
    url = "https://api.tavily.com/search"
    headers = {
        "Authorization": f"Bearer {TAVILY_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "query": query,
        "num": num_results,
        "search_settings": {
            "country": "us"
        }
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json().get("results", [])
    else:
        st.error(f"Tavily API Error: {response.text}")
        return []

def check_relevance(llm, keyword, title, snippet):
    prompt = (
        f"Given the search keyword: '{keyword}', "
        f"and the following website result:\n"
        f"Title: {title}\n"
        f"Snippet: {snippet}\n"
        "Is this website relevant to the search keyword? "
        "Reply with 'Yes' or 'No' and a short reason."
    )
    result = llm.invoke(prompt)
    return result.content.strip()

def get_keyword_data(keyword, match_type, display_limit):
    base_url = "https://api.semrush.com/"
    params = {
        "type": match_type,
        "key": SEMRUSH_API_KEY,
        "phrase": keyword,
        "database": "us",
        "export_columns": "Ph,Nq,Kd,In",  # In is Intent
        "display_limit": display_limit
    }
    response = requests.get(base_url, params=params)
    results = []
    if response.status_code == 200 and response.text.strip():
        lines = response.text.strip().split("\n")
        if len(lines) > 1:
            for line in lines[1:]:
                values = line.split(";")
                if len(values) == 4:
                    ph, nq, kd, intent = values
                    results.append({
                        "Keyword": ph,
                        "Search Volume": nq if nq.isdigit() else "No result",
                        "Keyword Difficulty Index": kd if kd.replace('.', '', 1).isdigit() else "No result",
                        "Intent": intent
                    })
    if not results:
        results.append({
            "Keyword": keyword,
            "Search Volume": "No result",
            "Keyword Difficulty Index": "No result",
            "Intent": "No result"
        })
    return results

# --- Streamlit App ---

st.set_page_config(page_title="SEO Keyword Toolkit", layout="wide")

st.title("🔑 SEO Keyword Toolkit")

tab1, tab2, tab3 = st.tabs([
    "1️⃣ Keyword Extractor (Azure OpenAI)",
    "2️⃣ Keyword Volume, Difficulty & Intent Categorizer (SEMrush API)",
    "3️⃣ Top 10 Website Search & Relevance Checker"
])

# --- Tab 1: Keyword Extractor ---
with tab1:
    st.header("Keyword Extractor with Azure OpenAI (GPT-4.1)")
    content = st.text_area("Paste your content here:", height=200)

    if st.button("Extract Keywords", key="extract_keywords"):
        if not content.strip():
            st.warning("Please enter some content.")
        else:
            # First pass: extract keywords
            prompt1 = (
                "Extract a list of SEO keywords and search phrases that a potential customer in the **USA** might use when searching for problems, solutions, or websites related to the following content. "
                "Think like a customer based in the United States: what would they type into Google to find this information? "
                "Return only the keywords and search phrases, one per line, no explanations or numbering.\n\n"
                f"Content:\n{content}"
            )

            with st.spinner("Extracting keywords..."):
                response1 = llm.invoke(prompt1)
                raw_keywords = response1.content.strip()

            # Second pass: clean and deduplicate
            prompt2 = (
                "Given the following list of keywords, return a cleaned, deduplicated list. "
                "Remove any generic, repetitive, or boilerplate phrases. "
                "Keep only unique, specific, high-value keywords. "
                "Return only the keywords, one per line, no explanations or numbering.\n\n"
                f"Keywords:\n{raw_keywords}"
            )
            with st.spinner("Cleaning keywords..."):
                response2 = llm.invoke(prompt2)
                keywords_final = response2.content.strip()

            st.success("Keywords extracted!")
            st.text_area("Keywords:", keywords_final, height=200)
            st.download_button(label="Download Keywords as .txt",
                               data=keywords_final,
                               file_name="keywords.txt",
                               mime="text/plain")

    st.markdown("---")
    st.subheader("🔍 Keyword Volume & Relevance Checker (SEMrush + Tavily)")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_kw_file = st.file_uploader("Upload a .txt file with keywords (one per line)", type="txt", key="tab1_kw_upload")
    with col2:
        pasted_kw_text = st.text_area("Or paste keywords here (one per line):", height=150, key="tab1_kw_paste")

    # Match type selection
    match_type_option = st.selectbox(
        "Select SEMrush Match Type",
        options=["EXACT_MATCH", "BROAD_MATCH"],
        format_func=lambda x: "Exact Match" if x == "EXACT_MATCH" else "Broad Match",
        key="tab1_match_type"
    )
    if match_type_option == "EXACT_MATCH":
        match_type = "phrase_this"
        display_limit = 1
    else:
        match_type = "phrase_related"
        display_limit = 10

    # Collect keywords from either source
    tab1_keywords = []
    if pasted_kw_text.strip():
        tab1_keywords = [kw.strip() for kw in pasted_kw_text.strip().splitlines() if kw.strip()]
    elif uploaded_kw_file is not None:
        tab1_keywords = uploaded_kw_file.read().decode("utf-8").splitlines()
        tab1_keywords = [kw.strip() for kw in tab1_keywords if kw.strip()]

    if tab1_keywords:
        st.write(f"**Total Keywords Provided:** {len(tab1_keywords)}")

        if st.button("Generate Keyword Analysis", key="tab1_generate_analysis"):
            analysis_results = []
            with st.spinner("Analyzing keywords with SEMrush and Tavily..."):
                for kw in tab1_keywords:
                    if match_type == "phrase_this":
                        # Exact match: only check the main keyword
                        semrush_data = get_keyword_data(kw, match_type, display_limit)
                        if semrush_data and semrush_data[0]["Search Volume"] != "No result":
                            try:
                                volume = int(semrush_data[0]["Search Volume"])
                            except:
                                volume = 0
                        else:
                            volume = 0

                        if volume >= 100:
                            tavily_results = tavily_search(kw, num_results=10)
                            if tavily_results:
                                for res in tavily_results:
                                    title = res.get("title", "")
                                    url = res.get("url", "")
                                    snippet = res.get("description", "")
                                    relevance = check_relevance(llm, kw, title, snippet)
                                    analysis_results.append({
                                        "Keyword": kw,
                                        "Search Volume (US)": str(volume),
                                        "Title": title,
                                        "URL": url,
                                        "Snippet": snippet,
                                        "Relevance": relevance
                                    })
                            else:
                                analysis_results.append({
                                    "Keyword": kw,
                                    "Search Volume (US)": str(volume),
                                    "Title": "",
                                    "URL": "",
                                    "Snippet": "",
                                    "Relevance": "No Tavily results"
                                })
                        else:
                            analysis_results.append({
                                "Keyword": kw,
                                "Search Volume (US)": str(volume) if volume > 0 else "No result",
                                "Title": "",
                                "URL": "",
                                "Snippet": "",
                                "Relevance": "Volume < 100 or No SEMrush result"
                            })
                    else:
                        # Broad match: check up to 10 related keywords
                        semrush_data_list = get_keyword_data(kw, match_type, display_limit)
                        found_any = False
                        for semrush_data in semrush_data_list:
                            rel_kw = semrush_data["Keyword"]
                            if semrush_data["Search Volume"] != "No result":
                                try:
                                    volume = int(semrush_data["Search Volume"])
                                except:
                                    volume = 0
                            else:
                                volume = 0

                            if volume >= 100:
                                found_any = True
                                tavily_results = tavily_search(rel_kw, num_results=10)
                                if tavily_results:
                                    for res in tavily_results:
                                        title = res.get("title", "")
                                        url = res.get("url", "")
                                        snippet = res.get("description", "")
                                        relevance = check_relevance(llm, rel_kw, title, snippet)
                                        analysis_results.append({
                                            "Input Keyword": kw,
                                            "Related Keyword": rel_kw,
                                            "Search Volume (US)": str(volume),
                                            "Title": title,
                                            "URL": url,
                                            "Snippet": snippet,
                                            "Relevance": relevance
                                        })
                                else:
                                    analysis_results.append({
                                        "Input Keyword": kw,
                                        "Related Keyword": rel_kw,
                                        "Search Volume (US)": str(volume),
                                        "Title": "",
                                        "URL": "",
                                        "Snippet": "",
                                        "Relevance": "No Tavily results"
                                    })
                            else:
                                analysis_results.append({
                                    "Input Keyword": kw,
                                    "Related Keyword": rel_kw,
                                    "Search Volume (US)": str(volume) if volume > 0 else "No result",
                                    "Title": "",
                                    "URL": "",
                                    "Snippet": "",
                                    "Relevance": "Volume < 100 or No SEMrush result"
                                })
                        if not semrush_data_list:
                            analysis_results.append({
                                "Input Keyword": kw,
                                "Related Keyword": "",
                                "Search Volume (US)": "No result",
                                "Title": "",
                                "URL": "",
                                "Snippet": "",
                                "Relevance": "No SEMrush result"
                            })

            if analysis_results:
                df_analysis = pd.DataFrame(analysis_results)
                st.dataframe(df_analysis)
                st.download_button(
                    label="Download Analysis as CSV",
                    data=df_analysis.to_csv(index=False).encode("utf-8"),
                    file_name="keyword_analysis.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No results found for any keyword.")

# --- Tab 2: Keyword Volume, Difficulty & Intent Categorizer (SEMrush API) ---
with tab2:
    st.header("Keyword Volume, Difficulty & Intent Categorizer (SEMrush API)")

    # Dropdown for match type
    match_type_option = st.selectbox(
        "Select Match Type",
        options=["EXACT_MATCH", "BROAD_MATCH"],
        format_func=lambda x: "Exact Match" if x == "EXACT_MATCH" else "Broad Match"
    )
    if match_type_option == "EXACT_MATCH":
        match_type = "phrase_this"
        display_limit = 1
    else:
        match_type = "phrase_related"
        display_limit = 10

    st.write("**You can either upload a .txt file or paste keywords below (one per line):**")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Upload a .txt file", type="txt", key="semrush_upload_new")
    with col2:
        text_input = st.text_area("Or paste keywords here", height=200, key="semrush_paste_new")

    # Collect keywords from either source
    keywords = []
    if uploaded_file is not None:
        keywords = uploaded_file.read().decode("utf-8").splitlines()
        keywords = [kw.strip() for kw in keywords if kw.strip()]
    elif text_input.strip():
        keywords = [kw.strip() for kw in text_input.strip().splitlines() if kw.strip()]

    if keywords:
        st.write(f"**Total Keywords:** {len(keywords)}")

        # Add Fetch Data button
        if st.button("Fetch Data", key="semrush_fetch_data"):
            cat1, cat2, cat3 = [], [], []
            no_result_count = 0

            with st.spinner("Fetching data from SEMrush..."):
                for keyword in keywords:
                    data_list = get_keyword_data(keyword, match_type, display_limit)
                    for data in data_list:
                        if data["Search Volume"] == "No result":
                            no_result_count += 1
                            cat1.append(data)  # Place all "No result" in cat1 for visibility
                            continue
                        # Use int() for comparison, but keep as string in DataFrame
                        try:
                            volume = int(data["Search Volume"])
                        except:
                            volume = 0
                        if volume <= 100:
                            cat1.append(data)
                        elif 100 < volume <= 1000:
                            cat2.append(data)
                        else:
                            cat3.append(data)

            # Convert to DataFrames
            df_cat1 = pd.DataFrame(cat1)
            df_cat2 = pd.DataFrame(cat2)
            df_cat3 = pd.DataFrame(cat3)

            # --- Ensure Arrow compatibility: convert all columns to string ---
            for df in [df_cat1, df_cat2, df_cat3]:
                for col in ["Search Volume", "Keyword Difficulty Index", "Intent"]:
                    if col in df.columns:
                        df[col] = df[col].astype(str)

            # Intent legend
            st.markdown("**Intent legend:** 0 - Commercial, 1 - Informational, 2 - Navigational, 3 - Transactional.")

            # Show DataFrames in Streamlit
            st.subheader("Category 1: Volume 0-100")
            st.dataframe(df_cat1)
            st.download_button(
                label="Download Cat-1 (0-100) as CSV",
                data=df_cat1.to_csv(index=False).encode('utf-8'),
                file_name="cat1_0-100.csv",
                mime="text/csv"
            )

            st.subheader("Category 2: Volume 101-1000")
            st.dataframe(df_cat2)
            st.download_button(
                label="Download Cat-2 (101-1000) as CSV",
                data=df_cat2.to_csv(index=False).encode('utf-8'),
                file_name="cat2_101-1000.csv",
                mime="text/csv"
            )

            st.subheader("Category 3: Volume 1001+")
            st.dataframe(df_cat3)
            st.download_button(
                label="Download Cat-3 (1001+) as CSV",
                data=df_cat3.to_csv(index=False).encode('utf-8'),
                file_name="cat3_1001plus.csv",
                mime="text/csv"
            )

            # Prepare Excel in memory
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_cat1.to_excel(writer, sheet_name='Cat-1 (0-100)', index=False)
                df_cat2.to_excel(writer, sheet_name='Cat-2 (101-1000)', index=False)
                df_cat3.to_excel(writer, sheet_name='Cat-3 (1001+)', index=False)
            output.seek(0)

            st.success("Excel file is ready!")
            st.download_button(
                label="Download Categorized Excel",
                data=output,
                file_name="categorized_keywords.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            if no_result_count == len(keywords):
                st.error("No results were found for any of the keywords. Please check your keywords or try a different match type.")
            elif no_result_count > 0:
                st.warning(f"No results were found for {no_result_count} keyword(s). They are marked as 'No result' in the table.")

    else:
        st.info("Please upload a file or paste keywords to begin.")

# --- Tab 3: Top 10 Website Search & Relevance Checker ---
with tab3:
    st.header("Top 10 Website Search & Relevance Checker")
    st.write("**You can either upload a .txt file or paste keywords below (one per line). If both are provided, pasted keywords will be used.**")
    uploaded_file = st.file_uploader("Upload a .txt file with your search keywords (one per line)", type=["txt"], key="tavily_upload")
    pasted_keywords = st.text_area("Or paste your keywords here (one per line):", height=150, key="tavily_paste")

    # Determine which input to use
    keywords = []
    if pasted_keywords.strip():
        keywords = [line.strip() for line in pasted_keywords.strip().splitlines() if line.strip()]
    elif uploaded_file is not None:
        keywords = [line.strip() for line in uploaded_file.read().decode("utf-8").splitlines() if line.strip()]

    if keywords:
        st.write(f"**Found {len(keywords)} keywords:**")
        st.write(keywords)

        if st.button("Search and Analyze", key="search_and_analyze"):
            all_results = []
            for keyword in keywords:
                if len(keyword) > 400:
                    st.warning(f"Keyword too long (>{len(keyword)} chars): {keyword[:50]}...")
                    continue
                st.info(f"Searching for: {keyword}")
                with st.spinner(f"Searching the web for '{keyword}'..."):
                    results = tavily_search(keyword, num_results=10)
                if results:
                    for res in results:
                        title = res.get("title", "")
                        url = res.get("url", "")
                        snippet = res.get("description", "")
                        relevance = check_relevance(llm, keyword, title, snippet)
                        all_results.append({
                            "Keyword": keyword,
                            "Title": title,
                            "URL": url,
                            "Snippet": snippet,
                            "Relevance": relevance
                        })
                else:
                    st.warning(f"No results found for '{keyword}' or API error.")

            if all_results:
                df = pd.DataFrame(all_results)
                st.dataframe(df)

                # Download CSV
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download results as CSV",
                    data=csv,
                    file_name="all_keywords_top10_results.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No results found for any keyword or API error.")