import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import csv
from datetime import datetime
import io
import base64
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import tempfile
import os


class ExportManager:
    def __init__(self):
        self.supported_formats = ['csv', 'json', 'excel', 'pdf', 'html']
    
    def export_data(self, data: Union[pd.DataFrame, Dict, List], 
                   filepath: str, format: str = 'csv', **kwargs) -> bool:
        
        format = format.lower()
        
        if format not in self.supported_formats:
            raise ValueError(f"Unsupported format. Choose from: {self.supported_formats}")
        
        try:
            if format == 'csv':
                return self._export_csv(data, filepath, **kwargs)
            elif format == 'json':
                return self._export_json(data, filepath, **kwargs)
            elif format == 'excel':
                return self._export_excel(data, filepath, **kwargs)
            elif format == 'pdf':
                return self._export_pdf(data, filepath, **kwargs)
            elif format == 'html':
                return self._export_html(data, filepath, **kwargs)
        except Exception as e:
            print(f"Export failed: {e}")
            return False
        
        return False
    
    def _export_csv(self, data: Union[pd.DataFrame, Dict, List], 
                   filepath: str, **kwargs) -> bool:
        
        df = self._ensure_dataframe(data)
        
        csv_kwargs = {
            'index': kwargs.get('include_index', True),
            'encoding': kwargs.get('encoding', 'utf-8'),
            'sep': kwargs.get('separator', ',')
        }
        
        df.to_csv(filepath, **csv_kwargs)
        return True
    
    def _export_json(self, data: Union[pd.DataFrame, Dict, List], 
                    filepath: str, **kwargs) -> bool:
        
        if isinstance(data, pd.DataFrame):
            json_data = data.to_dict(orient=kwargs.get('orient', 'records'))
        elif isinstance(data, (dict, list)):
            json_data = data
        else:
            json_data = str(data)
        
        json_kwargs = {
            'indent': kwargs.get('indent', 2),
            'ensure_ascii': kwargs.get('ensure_ascii', False)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, **json_kwargs, default=self._json_serializer)
        
        return True
    
    def _export_excel(self, data: Union[pd.DataFrame, Dict, List], 
                     filepath: str, **kwargs) -> bool:
        
        df = self._ensure_dataframe(data)
        
        excel_kwargs = {
            'index': kwargs.get('include_index', True),
            'engine': 'openpyxl'
        }
        
        with pd.ExcelWriter(filepath, **excel_kwargs) as writer:
            sheet_name = kwargs.get('sheet_name', 'Data')
            df.to_excel(writer, sheet_name=sheet_name, index=excel_kwargs['index'])
            
            if kwargs.get('auto_adjust_columns', True):
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = 0
                    column_name = column[0].column_letter
                    
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_name].width = adjusted_width
        
        return True
    
    def _export_pdf(self, data: Union[pd.DataFrame, Dict, List], 
                   filepath: str, **kwargs) -> bool:
        
        df = self._ensure_dataframe(data)
        
        doc = SimpleDocTemplate(filepath, pagesize=A4)
        elements = []
        styles = getSampleStyleSheet()
        
        title = kwargs.get('title', 'Cryptocurrency Data Export')
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            alignment=1
        )
        elements.append(Paragraph(title, title_style))
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        elements.append(Paragraph(f"Generated on: {timestamp}", styles['Normal']))
        elements.append(Spacer(1, 12))
        
        if not df.empty:
            max_rows = kwargs.get('max_rows', 50)
            if len(df) > max_rows:
                df_subset = df.head(max_rows)
                elements.append(Paragraph(f"Showing first {max_rows} rows of {len(df)} total rows", 
                                        styles['Normal']))
            else:
                df_subset = df
            
            table_data = []
            
            if kwargs.get('include_index', True):
                headers = ['Index'] + list(df_subset.columns)
                table_data.append(headers)
                
                for idx, row in df_subset.iterrows():
                    row_data = [str(idx)] + [str(val) for val in row.values]
                    table_data.append(row_data)
            else:
                headers = list(df_subset.columns)
                table_data.append(headers)
                
                for _, row in df_subset.iterrows():
                    row_data = [str(val) for val in row.values]
                    table_data.append(row_data)
            
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(table)
        
        doc.build(elements)
        return True
    
    def _export_html(self, data: Union[pd.DataFrame, Dict, List], 
                    filepath: str, **kwargs) -> bool:
        
        df = self._ensure_dataframe(data)
        
        title = kwargs.get('title', 'Cryptocurrency Data Export')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; text-align: center; }}
                .timestamp {{ text-align: center; color: #666; margin-bottom: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .summary {{ background-color: #e9f4ff; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <div class="timestamp">Generated on: {timestamp}</div>
        """
        
        if not df.empty:
            summary_info = f"""
            <div class="summary">
                <h3>Summary</h3>
                <p>Total Records: {len(df)}</p>
                <p>Columns: {len(df.columns)}</p>
                <p>Date Range: {df.index.min() if hasattr(df.index, 'min') else 'N/A'} to {df.index.max() if hasattr(df.index, 'max') else 'N/A'}</p>
            </div>
            """
            html_template += summary_info
        
        html_template += df.to_html(classes='data-table', table_id='data-table')
        html_template += """
        </body>
        </html>
        """
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        return True
    
    def _ensure_dataframe(self, data: Union[pd.DataFrame, Dict, List]) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, dict):
            return pd.DataFrame([data])
        elif isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                return pd.DataFrame(data)
            else:
                return pd.DataFrame(data, columns=['Value'])
        else:
            return pd.DataFrame([{'Value': str(data)}])
    
    def _json_serializer(self, obj):
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        
        return str(obj)
    
    def export_chart(self, fig: go.Figure, filepath: str, 
                    format: str = 'png', **kwargs) -> bool:
        
        format = format.lower()
        supported_chart_formats = ['png', 'jpg', 'jpeg', 'svg', 'pdf', 'html']
        
        if format not in supported_chart_formats:
            raise ValueError(f"Unsupported chart format. Choose from: {supported_chart_formats}")
        
        try:
            if format == 'html':
                fig.write_html(filepath, **kwargs)
            elif format == 'svg':
                fig.write_image(filepath, format='svg', **kwargs)
            elif format == 'pdf':
                fig.write_image(filepath, format='pdf', **kwargs)
            else:
                width = kwargs.get('width', 1200)
                height = kwargs.get('height', 800)
                fig.write_image(filepath, format=format, width=width, height=height)
            
            return True
        except Exception as e:
            print(f"Chart export failed: {e}")
            return False
    
    def create_report(self, data_dict: Dict[str, Any], filepath: str, 
                     title: str = "Cryptocurrency Analysis Report") -> bool:
        
        try:
            doc = SimpleDocTemplate(filepath, pagesize=A4)
            elements = []
            styles = getSampleStyleSheet()
            
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                alignment=1
            )
            elements.append(Paragraph(title, title_style))
            
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            elements.append(Paragraph(f"Generated on: {timestamp}", styles['Normal']))
            elements.append(Spacer(1, 20))
            
            for section_name, section_data in data_dict.items():
                elements.append(Paragraph(section_name.replace('_', ' ').title(), styles['Heading2']))
                elements.append(Spacer(1, 12))
                
                if isinstance(section_data, pd.DataFrame):
                    if not section_data.empty:
                        table_data = [list(section_data.columns)]
                        for _, row in section_data.head(10).iterrows():
                            table_data.append([str(val) for val in row.values])
                        
                        table = Table(table_data)
                        table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), 9),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                            ('FONTSIZE', (0, 1), (-1, -1), 8),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black)
                        ]))
                        
                        elements.append(table)
                
                elif isinstance(section_data, dict):
                    for key, value in section_data.items():
                        elements.append(Paragraph(f"<b>{key}:</b> {value}", styles['Normal']))
                
                elif isinstance(section_data, (list, str, int, float)):
                    elements.append(Paragraph(str(section_data), styles['Normal']))
                
                elements.append(Spacer(1, 20))
            
            doc.build(elements)
            return True
            
        except Exception as e:
            print(f"Report creation failed: {e}")
            return False
    
    def batch_export(self, export_jobs: List[Dict[str, Any]]) -> Dict[str, bool]:
        results = {}
        
        for job in export_jobs:
            job_name = job.get('name', f'export_{len(results)}')
            
            try:
                if 'chart' in job:
                    success = self.export_chart(
                        job['chart'], 
                        job['filepath'], 
                        job.get('format', 'png'),
                        **job.get('kwargs', {})
                    )
                else:
                    success = self.export_data(
                        job['data'],
                        job['filepath'],
                        job.get('format', 'csv'),
                        **job.get('kwargs', {})
                    )
                
                results[job_name] = success
                
            except Exception as e:
                print(f"Batch export job '{job_name}' failed: {e}")
                results[job_name] = False
        
        return results
    
    def get_export_summary(self, data: Union[pd.DataFrame, Dict, List]) -> Dict[str, Any]:
        df = self._ensure_dataframe(data)
        
        summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'column_names': list(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'data_types': df.dtypes.to_dict(),
            'has_missing_values': df.isnull().any().any(),
            'missing_values_per_column': df.isnull().sum().to_dict()
        }
        
        if hasattr(df.index, 'min') and hasattr(df.index, 'max'):
            summary['index_range'] = {
                'min': df.index.min(),
                'max': df.index.max()
            }
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            summary['numeric_summary'] = df[numeric_columns].describe().to_dict()
        
        return summary