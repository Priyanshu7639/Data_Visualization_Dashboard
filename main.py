import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import io
from scipy import stats
from viz_assistant import VizAssistant
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DataAnalyzer:
    def __init__(self):
        self.df = None
        self.processed_df = None
        self.viz_assistant = VizAssistant(os.getenv('GOOGLE_API_KEY'))
        self.current_viz_settings = {
            'x': None,
            'y': None,
            'color': None,
            'title': None,
            'size': None,
            'trendline': None
        }
    
    def load_data(self, file):
        """Load data from various file formats"""
        try:
            if file.name.endswith('.csv'):
                self.df = pd.read_csv(file)
            elif file.name.endswith(('.xls', '.xlsx')):
                self.df = pd.read_excel(file)
            self.processed_df = self.df.copy()
            return True
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return False

    def preprocess_data(self, options):
        """Apply selected preprocessing steps"""
        try:
            if 'remove_duplicates' in options:
                self.processed_df = self.processed_df.drop_duplicates()

            if 'handle_missing' in options:
                strategy = st.selectbox('Missing Value Strategy', 
                                      ['mean', 'median', 'most_frequent', 'constant'])
                numeric_columns = self.processed_df.select_dtypes(include=[np.number]).columns
                if numeric_columns.any():
                    imputer = SimpleImputer(strategy=strategy)
                    self.processed_df[numeric_columns] = imputer.fit_transform(
                        self.processed_df[numeric_columns]
                    )

            if 'scale_features' in options:
                scaling_method = st.selectbox('Scaling Method', 
                                            ['standard', 'minmax'])
                numeric_columns = self.processed_df.select_dtypes(include=[np.number]).columns
                if numeric_columns.any():
                    if scaling_method == 'standard':
                        scaler = StandardScaler()
                    else:
                        scaler = MinMaxScaler()
                    self.processed_df[numeric_columns] = scaler.fit_transform(
                        self.processed_df[numeric_columns]
                    )

            return True
        except Exception as e:
            st.error(f"Error in preprocessing: {str(e)}")
            return False

    def update_viz_settings(self, command: str) -> dict:
        """Update visualization settings based on natural language command"""
        try:
            # Simple command processing logic
            command = command.lower()
            new_settings = self.current_viz_settings.copy()
            
            if 'bigger' in command or 'larger' in command:
                new_settings['size'] = 10
            
            if 'trend' in command or 'trendline' in command:
                new_settings['trendline'] = 'ols'
            
            if 'title' in command:
                # Extract title between quotes if present
                import re
                title_match = re.search(r"'(.*?)'", command)
                if title_match:
                    new_settings['title'] = title_match.group(1)
            
            return new_settings
            
        except Exception as e:
            st.error(f"Error processing command: {str(e)}")
            return self.current_viz_settings

    def create_visualization(self, viz_type, settings):
        """Create various types of visualizations"""
        try:
            if viz_type == "Scatter Plot":
                fig = px.scatter(
                    self.processed_df, 
                    x=settings['x'], 
                    y=settings['y'],
                    color=settings.get('color'),
                    title=f"{settings['y']} vs {settings['x']}",
                    trendline=settings.get('trendline'),
                    marginal_x='histogram',
                    marginal_y='histogram'
                )

            elif viz_type == "Pair Plot":
                numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns
                fig = px.scatter_matrix(
                    self.processed_df,
                    dimensions=numeric_cols,
                    color=settings.get('color'),
                    title="Pair Plot"
                )

            elif viz_type == "Distribution Plot":
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=self.processed_df[settings['x']],
                    name='Histogram',
                    nbinsx=30
                ))
                fig.add_trace(go.Violin(
                    x=self.processed_df[settings['x']],
                    name='Violin',
                    side='positive'
                ))
                fig.update_layout(title=f"Distribution of {settings['x']}")

            elif viz_type == "Box Plot":
                fig = px.box(
                    self.processed_df, 
                    x=settings['x'], 
                    y=settings['y'],
                    color=settings.get('color'),
                    notched=True,
                    title=f"Box Plot of {settings['y']} by {settings['x']}"
                )

            elif viz_type == "Violin Plot":
                fig = px.violin(
                    self.processed_df,
                    x=settings['x'],
                    y=settings['y'],
                    color=settings.get('color'),
                    box=True,
                    points="all",
                    title=f"Violin Plot of {settings['y']} by {settings['x']}"
                )

            elif viz_type == "Correlation Heatmap":
                numeric_df = self.processed_df.select_dtypes(include=[np.number])
                corr = numeric_df.corr()
                mask = np.triu(np.ones_like(corr, dtype=bool))
                fig = px.imshow(
                    corr,
                    title="Correlation Heatmap",
                    color_continuous_scale='RdBu_r',
                    aspect='auto'
                )

            elif viz_type == "Time Series":
                fig = px.line(
                    self.processed_df,
                    x=settings['x'],
                    y=settings['y'],
                    color=settings.get('color'),
                    title=f"Time Series of {settings['y']}",
                )
                fig.update_xaxes(rangeslider_visible=True)

            elif viz_type == "3D Scatter":
                fig = px.scatter_3d(
                    self.processed_df,
                    x=settings['x'],
                    y=settings['y'],
                    z=settings['z'],
                    color=settings.get('color'),
                    title="3D Scatter Plot"
                )

            elif viz_type == "Parallel Coordinates":
                numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns
                fig = px.parallel_coordinates(
                    self.processed_df,
                    dimensions=numeric_cols,
                    color=settings.get('color'),
                    title="Parallel Coordinates Plot"
                )

            elif viz_type == "QQ Plot":
                fig = go.Figure()
                qq = stats.probplot(self.processed_df[settings['x']], dist="norm")
                fig.add_scatter(x=qq[0][0], y=qq[0][1], mode='markers')
                fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][0], mode='lines'))
                fig.update_layout(title=f"Q-Q Plot of {settings['x']}")

            elif viz_type == "Missing Values Heatmap":
                missing = self.processed_df.isnull()
                fig = px.imshow(
                    missing,
                    title="Missing Values Heatmap",
                    color_continuous_scale=['white', 'red']
                )

            return fig
        except Exception as e:
            st.error(f"Error in visualization: {str(e)}")
            return None

    def create_statistical_summary(self, column):
        """Create statistical summary for a numeric column"""
        stats_dict = {
            'Mean': self.processed_df[column].mean(),
            'Median': self.processed_df[column].median(),
            'Std Dev': self.processed_df[column].std(),
            'Skewness': self.processed_df[column].skew(),
            'Kurtosis': self.processed_df[column].kurtosis(),
            'IQR': self.processed_df[column].quantile(0.75) - self.processed_df[column].quantile(0.25)
        }
        return pd.Series(stats_dict)

def main():
    # Add API key check
    if not os.getenv('GOOGLE_API_KEY'):
        st.error("Please set your GOOGLE_API_KEY in the .env file")
        return
        
    st.title("Advanced Data Analysis Dashboard")
    
    # Initialize analyzer
    analyzer = DataAnalyzer()
    
    # Sidebar for file upload and preprocessing
    with st.sidebar:
        st.header("Data Input")
        uploaded_file = st.file_uploader("Upload your data", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None and analyzer.load_data(uploaded_file):
            st.header("Preprocessing Options")
            preprocessing_options = st.multiselect(
                "Select preprocessing steps",
                ['remove_duplicates', 'handle_missing', 'scale_features']
            )
            
            if preprocessing_options:
                analyzer.preprocess_data(preprocessing_options)
    
    # Main content area
    if analyzer.processed_df is not None:
        # Data Overview
        st.header("Data Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", analyzer.processed_df.shape[0])
        with col2:
            st.metric("Columns", analyzer.processed_df.shape[1])
        with col3:
            st.metric("Missing Values", analyzer.processed_df.isna().sum().sum())
        
        # Data Preview
        with st.expander("Show Data Preview"):
            st.dataframe(analyzer.processed_df.head())
            
        # Data Info
        with st.expander("Show Data Info"):
            buffer = io.StringIO()
            analyzer.processed_df.info(buf=buffer)
            st.text(buffer.getvalue())
        
        # Visualization section
        st.header("📊 Visualization")
        viz_type = st.selectbox(
            "Select Visualization Type",
            ["Scatter Plot", "Line Plot", "Bar Chart", "Box Plot", 
             "Violin Plot", "Histogram", "Correlation Heatmap"]
        )

        # Basic settings selection
        columns = analyzer.processed_df.columns.tolist()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            analyzer.current_viz_settings['x'] = st.selectbox("Select X axis", columns)
        with col2:
            if viz_type not in ["Histogram"]:
                analyzer.current_viz_settings['y'] = st.selectbox("Select Y axis", columns)
        with col3:
            analyzer.current_viz_settings['color'] = st.selectbox(
                "Color by", ['None'] + columns
            )
            if analyzer.current_viz_settings['color'] == 'None':
                analyzer.current_viz_settings['color'] = None

        # Create visualization with current settings
        fig = analyzer.create_visualization(viz_type, analyzer.current_viz_settings)
        if fig:
            st.plotly_chart(fig)

            # Show current settings
            with st.expander("⚙️ Current Visualization Settings"):
                st.json(analyzer.current_viz_settings)

        # AI Command Interface (moved below visualization)
        st.header("🤖 AI Visualization Assistant")
        command_col1, command_col2 = st.columns([3, 1])
        with command_col1:
            command = st.text_input(
                "Enter your visualization command",
                placeholder="e.g., 'make the points bigger' or 'add a trend line'"
            )
        with command_col2:
            if st.button("Apply Changes"):
                if command:
                    with st.spinner('Processing your command...'):
                        new_settings = analyzer.update_viz_settings(command)
                        analyzer.current_viz_settings.update(new_settings)
                        st.success("Changes applied!")
                        # Rerun visualization with new settings
                        fig = analyzer.create_visualization(viz_type, analyzer.current_viz_settings)
                        if fig:
                            st.plotly_chart(fig)

        # Show example commands
        with st.expander("📝 Example Commands"):
            st.markdown("""
            Try these commands:
            - "Make the scatter plot points larger"
            - "Add a trend line"
            - "Change title to 'Sales Analysis'"
            - "Make the axis labels bigger"
            - "Show confidence intervals"
            - "Add annotations for outliers"
            """)

if __name__ == "__main__":
    main()