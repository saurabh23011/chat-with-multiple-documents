import streamlit as st
import google.generativeai as genai
import pandas as pd
import plotly.express as px
import docx
import PyPDF2
import io
from PIL import Image
import os
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Gemini
api_key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')
vision_model = genai.GenerativeModel('gemini-1.5-flash')

# File Processing Functions
def read_text_file(file) -> str:
    """Read content from a text file"""
    return file.getvalue().decode('utf-8')

def read_docx(file) -> str:
    """Read content from a DOCX file"""
    doc = docx.Document(file)
    return ' '.join([paragraph.text for paragraph in doc.paragraphs])

def read_pdf(file) -> str:
    """Read content from a PDF file"""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def process_csv(file) -> Tuple[pd.DataFrame, str]:
    """Process CSV/Excel file and return dataframe and description"""
    df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
    description = f"DataFrame with {len(df)} rows and {len(df.columns)} columns. "
    description += f"Columns: {', '.join(df.columns)}"
    return df, description

def process_image(image_file) -> Image.Image:
    """Process uploaded image file"""
    return Image.open(image_file)

# Data Analysis Functions
def analyze_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze DataFrame to determine suitable visualization options"""
    analysis = {
        'numeric_columns': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist(),
        'row_count': len(df),
        'column_count': len(df.columns)
    }
    return analysis

def suggest_visualizations(df_analysis: Dict[str, Any]) -> str:
    """Generate visualization suggestions based on DataFrame analysis"""
    suggestions = []
    
    if df_analysis['numeric_columns']:
        suggestions.append(f"- Distribution plots for numeric columns: {', '.join(df_analysis['numeric_columns'])}")
        if len(df_analysis['numeric_columns']) >= 2:
            suggestions.append("- Scatter plots to show relationships between numeric variables")
    
    if df_analysis['categorical_columns']:
        suggestions.append(f"- Bar charts for categorical columns: {', '.join(df_analysis['categorical_columns'])}")
        if df_analysis['numeric_columns']:
            suggestions.append("- Box plots to show numeric distributions by category")
    
    if df_analysis['datetime_columns']:
        suggestions.append("- Time series plots for temporal analysis")
    
    return "\n".join(suggestions)

def generate_visualization_prompt(df: pd.DataFrame, query: str) -> str:
    """Generate a detailed prompt for visualization generation"""
    analysis = analyze_dataframe(df)
    
    prompt = f"""
    As a data visualization expert, create a Plotly Express visualization based on the following:
    
    DataFrame Information:
    - Numeric columns: {', '.join(analysis['numeric_columns'])}
    - Categorical columns: {', '.join(analysis['categorical_columns'])}
    - Datetime columns: {', '.join(analysis['datetime_columns'])}
    - Total rows: {analysis['row_count']}
    
    User Query: "{query}"
    
    Generate only the Plotly Express code that best represents the data according to the user's query.
    Only use actual column names from the provided DataFrame.
    
    The code must start with 'px.' and be in this exact format:
    px.scatter(df, x='actual_column_name', y='actual_column_name', title='Descriptive Title')
    OR
    px.line(df, x='actual_column_name', y='actual_column_name', title='Descriptive Title')
    OR
    px.bar(df, x='actual_column_name', y='actual_column_name', title='Descriptive Title')
    
    Include color only if it makes sense for the visualization.
    Use only existing column names from the DataFrame.
    Do not include any explanation, only the code.
    """
    return prompt

def perform_data_analysis(df: pd.DataFrame):
    """Perform basic data analysis"""
    st.subheader("Data Analysis")
    
    # Basic statistics
    st.write("Basic Statistics:")
    st.write(df.describe())
    
    # Missing values analysis
    st.write("Missing Values Analysis:")
    missing_data = df.isnull().sum()
    if missing_data.any():
        st.write(missing_data[missing_data > 0])
    else:
        st.write("No missing values found")
    
    # Correlation analysis for numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 1:
        st.write("Correlation Matrix:")
        correlation = df[numeric_cols].corr()
        fig = px.imshow(correlation, 
                       title="Correlation Heatmap",
                       labels=dict(x="Features", y="Features", color="Correlation"))
        st.plotly_chart(fig)

def create_visualization_interface(df: pd.DataFrame):
    """Create an interactive visualization interface in Streamlit"""
    st.subheader("Data Visualization")
    
    # Display basic data info
    st.write("Dataset Overview:")
    st.write(f"- Rows: {len(df)}")
    st.write(f"- Columns: {', '.join(df.columns)}")
    
    # Analysis and suggestions
    analysis = analyze_dataframe(df)
    st.write("Suggested Visualizations:")
    st.write(suggest_visualizations(analysis))
    
    # Visualization options
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Custom Query", "Quick Plots"]
    )
    
    if viz_type == "Custom Query":
        st.write("Available columns:", ", ".join(df.columns))
        viz_query = st.text_input(
            "Describe the visualization you want (e.g., 'Show me a scatter plot of sales vs profit')"
        )
        if viz_query:
            prompt = generate_visualization_prompt(df, viz_query)
            try:
                response = model.generate_content(prompt)
                viz_code = response.text.strip()
                viz_code = viz_code.replace('```python', '').replace('```', '').strip()
                
                st.code(viz_code, language='python')
                
                fig = eval(viz_code)
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"""Error generating visualization: {str(e)}
                \nTry using the Quick Plots option instead, or make sure to use exact column names: {', '.join(df.columns)}""")
    
    else:  # Quick Plots
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("X-axis", df.columns)
        with col2:
            y_axis = st.selectbox("Y-axis", df.columns)
        
        plot_type = st.selectbox(
            "Plot Type",
            ["Scatter", "Line", "Bar", "Box", "Histogram"]
        )
        
        color_by = st.selectbox("Color by (optional)", ["None"] + list(df.columns))
        
        try:
            if plot_type == "Scatter":
                fig = px.scatter(df, x=x_axis, y=y_axis, 
                               color=None if color_by == "None" else color_by,
                               title=f"{y_axis} vs {x_axis}")
            elif plot_type == "Line":
                fig = px.line(df, x=x_axis, y=y_axis,
                            color=None if color_by == "None" else color_by,
                            title=f"{y_axis} over {x_axis}")
            elif plot_type == "Bar":
                fig = px.bar(df, x=x_axis, y=y_axis,
                           color=None if color_by == "None" else color_by,
                           title=f"{y_axis} by {x_axis}")
            elif plot_type == "Box":
                fig = px.box(df, x=x_axis, y=y_axis,
                           color=None if color_by == "None" else color_by,
                           title=f"Distribution of {y_axis} by {x_axis}")
            elif plot_type == "Histogram":
                fig = px.histogram(df, x=x_axis,
                                 color=None if color_by == "None" else color_by,
                                 title=f"Distribution of {x_axis}")
            
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")

def main():
    st.title("Chat With Any File and Analysis App")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a document (.txt, .docx, .pdf, .csv, .xlsx, or image)",
        type=['txt', 'docx', 'pdf', 'csv', 'xlsx', 'png', 'jpg', 'jpeg']
    )
    
    if uploaded_file:
        try:
            # Handle data files (CSV/Excel)
            if uploaded_file.type in ["text/csv", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                df, content = process_csv(uploaded_file)
                
                # Create tabs for different functionalities
                tabs = st.tabs(["Data Preview", "Analysis", "Visualization", "Chat"])
                
                with tabs[0]:  # Data Preview
                    st.subheader("Data Preview")
                    st.dataframe(df)
                
                with tabs[1]:  # Analysis
                    perform_data_analysis(df)
                
                with tabs[2]:  # Visualization
                    create_visualization_interface(df)
                
                with tabs[3]:  # Chat
                    st.subheader("Chat with your Data")
                    chat_input = st.text_input("Ask a question about your data:")
                    if chat_input:
                        prompt = f"""
                        Context: DataFrame with columns: {', '.join(df.columns)}
                        Data Preview: {df.head().to_string()}
                        User Question: {chat_input}
                        Please provide a clear and concise response based on the data.
                        """
                        response = model.generate_content(prompt)
                        st.write("Response:", response.text)
                
                st.session_state['content'] = content
                st.session_state['df'] = df
            
            # Handle documents and images (PDF, DOCX, TXT, Images)
            else:
                if uploaded_file.type == "text/plain":
                    content = read_text_file(uploaded_file)
                    st.success("Text file processed successfully")
                    
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    content = read_docx(uploaded_file)
                    st.success("Word document processed successfully")
                    
                elif uploaded_file.type == "application/pdf":
                    content = read_pdf(uploaded_file)
                    st.success("PDF processed successfully")
                    
                elif uploaded_file.type in ["image/png", "image/jpeg"]:
                    image = process_image(uploaded_file)
                    st.image(image, caption="Uploaded Image")
                    content = "Image uploaded successfully"
                    st.session_state['image'] = image
                
                # Create tabs for document/image interface
                tabs = st.tabs(["Document", "Chat"])
                
                with tabs[0]:  # Document
                    st.subheader("Document Content")
                    if uploaded_file.type in ["image/png", "image/jpeg"]:
                        st.image(image, caption="Uploaded Image")
                    else:
                        st.write(content[:1000] + "..." if len(content) > 1000 else content)
                
                with tabs[1]:  # Chat
                    st.subheader("Chat with your Image")
                    question = st.text_input("Ask a question about the image:")
                    
                    if question:
                        try:
                            if 'image' in st.session_state:
                                response = vision_model.generate_content([question, st.session_state['image']])
                            else:
                                prompt = f"""
                                Context: {content}
                                Question: {question}
                                Please provide a clear and concise answer based on the context.
                                """
                                response = model.generate_content(prompt)
                            
                            st.write("Answer:", response.text)
                            
                        except Exception as e:
                            st.error(f"Error generating response: {str(e)}")
                
                st.session_state['content'] = content
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return

if __name__ == "__main__":
    main()