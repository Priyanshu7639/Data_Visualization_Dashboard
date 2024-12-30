# Advanced Data Visualization Dashboard 📊

An interactive data visualization tool with AI-powered customization capabilities. This application allows users to upload data, create various types of visualizations, and modify them using natural language commands.

## Features 🌟

### Data Handling
- Support for CSV and Excel file formats
- Automatic data type detection
- Basic data preprocessing options
- Missing value handling
- Feature scaling capabilities

### Visualizations
- Scatter Plots
- Line Plots
- Bar Charts
- Box Plots
- Violin Plots
- Histograms
- Correlation Heatmaps

### AI Assistant
- Natural language commands for visualization modification
- Interactive command interface
- Real-time visualization updates
- Example commands for easy reference

## Installation 🛠️

1. Clone the repository:
   ```bash
   git clone https://github.com/Priyanshu7639/Data_Visualization_Dashboard.git
   cd Data_Visualization_Dashboard
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage 💡

1. Start the application:
   ```bash
   streamlit run main.py
   ```

2. Upload your data:
   - Click "Browse files" in the sidebar
   - Select your CSV or Excel file

3. Preprocess your data (optional):
   - Remove duplicates
   - Handle missing values
   - Scale features

4. Create visualizations:
   - Select visualization type
   - Choose variables for axes
   - Add color grouping if needed

5. Modify visualizations using AI:
   - Enter commands in natural language
   - Click "Apply Changes"
   - See immediate updates to your visualization

## Example Commands 🗣️

- "Make the scatter plot points larger"
- "Add a trend line"
- "Change title to 'Sales Analysis'"
- "Make the axis labels bigger"
- "Show confidence intervals"
- "Add annotations for outliers"

## Dependencies 📚

- streamlit
- pandas
- numpy
- plotly
- seaborn
- matplotlib
- scikit-learn
- openpyxl

## Contributing 🤝

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙏

- Streamlit for the amazing web framework
- Plotly for interactive visualizations
- The open-source community for inspiration and support

### Additional Notes 📝

- Make sure you have Python 3.7+ installed
- The application works best with clean, structured data
- Large datasets might take longer to process
- Supported file formats: .csv, .xlsx, .xls

### Troubleshooting 🔧

1. If you encounter installation issues:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt --force-reinstall
   ```

2. If visualizations don't load:
   - Check your internet connection
   - Ensure data format is correct
   - Verify column names don't contain special characters

3. For memory issues with large datasets:
   - Try reducing the dataset size
   - Close other applications
   - Increase system swap space

### Future Enhancements 🚀

- Additional visualization types
- More AI command capabilities
- Advanced data preprocessing options
- Export functionality for modified visualizations
- Custom theme support
- Real-time collaboration features

For more information or support, please open an issue in the GitHub repository.
