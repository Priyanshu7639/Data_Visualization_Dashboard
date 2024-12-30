from typing import Dict, Any
import google.generativeai as genai
import json
import streamlit as st

class VizAssistant:
    def __init__(self, api_key: str):
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
        self.viz_options = {
            "scatter": ["x", "y", "color", "size", "title", "trendline"],
            "box": ["x", "y", "color", "title", "notched"],
            "line": ["x", "y", "color", "title", "markers"],
            "histogram": ["x", "bins", "title", "cumulative"],
            "heatmap": ["title", "colorscale", "annotations"],
            "3d_scatter": ["x", "y", "z", "color", "size", "title"],
            "violin": ["x", "y", "color", "title", "points"],
        }

    def process_command(self, command: str, current_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Process natural language command and return updated visualization settings"""
        try:
            # Create the prompt
            prompt = f"""
            Act as a data visualization expert. Convert the following natural language command 
            into visualization settings. Return only valid JSON with the updated settings.
            
            Available settings: x, y, z, color, size, title, trendline, bins, 
            colorscale, markers, annotations, points, notched.
            
            Current settings: {json.dumps(current_settings)}
            Command: {command}
            
            Return only the modified settings as JSON. Do not include any explanation or additional text.
            """

            # Get response from Gemini
            response = self.model.generate_content(prompt)
            
            # Extract JSON from response
            try:
                # Clean the response to ensure it only contains JSON
                response_text = response.text.strip()
                # Remove any markdown code block indicators if present
                response_text = response_text.replace('```json', '').replace('```', '').strip()
                new_settings = json.loads(response_text)
                return {**current_settings, **new_settings}
            except json.JSONDecodeError as e:
                st.error(f"Error parsing JSON response: {str(e)}")
                return current_settings
            
        except Exception as e:
            st.error(f"Error processing command: {str(e)}")
            return current_settings

    def get_visualization_suggestions(self, data_info: Dict[str, Any]) -> str:
        """Get AI suggestions for visualization improvements"""
        try:
            prompt = f"""
            As a data visualization expert, provide 3 specific suggestions to improve the current visualization.
            Consider the following data information:
            {json.dumps(data_info)}
            
            Provide concise, actionable suggestions.
            """
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"Error getting suggestions: {str(e)}" 