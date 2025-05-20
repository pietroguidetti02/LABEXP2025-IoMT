#!/usr/bin/env python3
import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.subplots as sp
import plotly.figure_factory as ff
import pandas as pd
from io import StringIO
import numpy as np
import os
import time
from datetime import datetime, timedelta
import json
import subprocess

# Configuration
UPDATE_INTERVAL = 300  # Update interval in milliseconds
MAX_POINTS = 20  # Maximum number of points to display in time series

# Get the latest data folder
def get_latest_data_folder():
    try:
        with open("latest_data_folder.txt", 'r') as f:
            folder = f.read().strip()
            if os.path.exists(folder):
                return folder
    except:
        pass
    
    # If file doesn't exist or folder doesn't exist, find the most recent folder
    data_folders = [f for f in os.listdir('.') if f.startswith('positions_data_')]
    if data_folders:
        latest_folder = max(data_folders)
        return latest_folder
    
    return None

# Get CSV file path
def get_csv_path():
    folder = get_latest_data_folder()
    if folder:
        csv_path = os.path.join(folder, 'wearable_data.csv')
        if os.path.exists(csv_path):
            return csv_path
    return None

# Define anchor positions (same as in the server)
'''
ANCHOR_POSITIONS = { #try
    'anchor1': (0, 0),
    'anchor2': (10, 0),
    'anchor3': (10, 10),
    'anchor4': (0, 10)
}'''
ANCHOR_POSITIONS = { #EP2
    'anchor1': (0, 0),
    'anchor2': (0, 7),
    'anchor3': (3.5, 7),
    'anchor4': (3.5, 0)
}
'''
ANCHOR_POSITIONS = { #home, livingroom
    'anchor1': (0, 0),
    'anchor2': (6.5, 0),
    'anchor3': (6, 4.3),
    'anchor4': (0.7, 6)
}'''

# Initialize the Dash app
app = dash.Dash(__name__, update_title=None)
app.title = "Enhanced Wearable Positioning System"

# Styles for tabs
tab_style = {
    'padding': '10px',
    'fontSize': '16px',
    'backgroundColor': '#f8f9fa',
    'borderBottom': '1px solid #dee2e6',
    'borderTop': '1px solid #dee2e6'
}

tab_selected_style = {
    'padding': '10px',
    'fontSize': '16px',
    'fontWeight': 'bold',
    'backgroundColor': '#e9ecef',
    'borderBottom': '1px solid #dee2e6',
    'borderTop': '3px solid #3498db'
}

card_style = {
    'flex': '1', 
    'backgroundColor': '#ecf0f1', 
    'borderRadius': '10px', 
    'padding': '15px', 
    'margin': '5px',
    'minWidth': '200px'
}

# Define the layout with tabs for organization
app.layout = html.Div([
    html.H1("Enhanced Wearable Positioning and Distance Tracking System", 
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
    html.H2("6-Minute Walk Test project",
            style={'textAlign': 'center', 'color': '#2c3e70', 'marginBottom': '3px'}),
    html.H3("IoMT 2025 | LABEXP",
            style={'textAlign': 'center', 'color': '#2c3e70', 'marginBottom': '3px'}),
    html.H4("Pietro Guidetti, Minglei Shao, Alice, Ameni, IO",
            style={'textAlign': 'center', 'color': '#2c3e90', 'marginBottom': '10px'}),

    #patient data block
    html.Div([
    html.H3("Patient Information", style={'textAlign': 'center'}),

    html.Div([
        html.Div([
            html.Label("First Name:"),
            dcc.Input(id='patient-name', type='text', placeholder='Mario', style={'width': '100%'})
        ], style={'flex': '1', 'margin': '10px'}),
        
        html.Div([
            html.Label("Last Name:"),
            dcc.Input(id='patient-surname', type='text', placeholder='Rossi', style={'width': '100%'})
        ], style={'flex': '1', 'margin': '10px'}),

        html.Div([
            html.Label("Age:"),
            dcc.Input(id='patient-age', type='number', placeholder='35', min=1, max=120, step=1, style={'width': '100%'})
        ], style={'flex': '1', 'margin': '10px'}),
        
        html.Div([
            html.Label("Height (m):"),
            dcc.Input(id='patient-height', type='number', placeholder='1.75', min=1.0, max=2.5, step=0.01, style={'width': '100%'})
        ], style={'flex': '1', 'margin': '10px'}),
        
        html.Div([
            html.Label("Gender:"),
            dcc.Dropdown(
                id='patient-gender',
                options=[
                    {'label': 'Male', 'value': 'male'},
                    {'label': 'Female', 'value': 'female'}
                ],
                value='male',
                style={'width': '100%'}
            )
        ], style={'flex': '1', 'margin': '10px'}),
    ], style={'display': 'flex', 'flexWrap': 'wrap'}),

    html.Button("Confirm Patient Data", id='confirm-patient-btn', n_clicks=0, style={'fontSize': '16px', 'marginTop': '10px'}),
    html.Div(id='patient-confirmation-msg'),

    #not sure this goes here (Nel layout (sotto i campi paziente), metti:)
    html.Button("Start New 6-Minute Walk Test", id="start-test-button", n_clicks=0, style={'fontSize': '18px'}),
    html.Div(id="test-status-msg"),
    dcc.Interval(id="check-process", interval=2000, n_intervals=0, disabled=True),
    dcc.Store(id="test-running", data=False),
]),


  
    # timer and button
    html.Div([
    html.Button("Start timer", id='start-button', n_clicks=0, disabled=False,
            style={'padding': '10px', 'fontSize': '18px', 'marginTop': '20px'}),
    html.Button('Stop', id='reset-button', n_clicks=0, style={'padding': '10px', 'fontSize': '18px', 'marginLeft': '10px'}),
    html.Div(id='timer-display', style={'fontSize': '24px', 'textAlign': 'center', 'marginTop': '10px'}),
    dcc.Interval(id='timer-interval', interval=1000, n_intervals=0, disabled=True),
    dcc.Store(id='start-time', data=None),
    dcc.Store(id='patient-data-valid', data=False),
    html.Div(id='data-warning-msg', style={'color': 'red', 'fontWeight': 'bold'})
], style={'margin': '20px', 'textAlign': 'center'}),


    # Statistics cards
    html.Div([
        html.Div([
            html.H4("Total Distance", style={'textAlign': 'center', 'margin': '5px'}),
            html.Div(id='total-distance', style={'textAlign': 'center', 'fontSize': '24px', 'fontWeight': 'bold'}),
            html.Div("meters", style={'textAlign': 'center', 'color': '#7f8c8d'})
        ], style=card_style),
        
        html.Div([
            html.H4("Current Speed", style={'textAlign': 'center', 'margin': '5px'}),
            html.Div(id='current-speed', style={'textAlign': 'center', 'fontSize': '24px', 'fontWeight': 'bold'}),
            html.Div("m/s", style={'textAlign': 'center', 'color': '#7f8c8d'})
        ], style=card_style),
        
        html.Div([
            html.H4("Total Steps", style={'textAlign': 'center', 'margin': '5px'}),
            html.Div(id='total-steps', style={'textAlign': 'center', 'fontSize': '24px', 'fontWeight': 'bold'}),
            html.Div("steps", style={'textAlign': 'center', 'color': '#7f8c8d'})
        ], style=card_style),
        
        html.Div([
            html.H4("Step Rate", style={'textAlign': 'center', 'margin': '5px'}),
            html.Div(id='step-rate', style={'textAlign': 'center', 'fontSize': '24px', 'fontWeight': 'bold'}),
            html.Div("steps/min", style={'textAlign': 'center', 'color': '#7f8c8d'})
        ], style=card_style),
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'margin': '10px 0', 'flexWrap': 'wrap'}),
    
    # Main content with tabs for organization
    dcc.Tabs([
        # Tab 1: Original Plots
        dcc.Tab(label="Main Tracking", children=[
            html.Div([
                # Original 2D Position plot
                html.Div([
                    html.H3("Real-time Position", style={'textAlign': 'center'}),
                    html.P("Mostra la posizione attuale e il percorso effettuato, aiutando a visualizzare il movimento nell'area monitorata.", 
                        style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '14px', 'paddingBottom': '10px'}),
                    dcc.Graph(
                        id='position-plot',
                        style={'height': '60vh'},
                        config={'displayModeBar': False}
                    ),
                ], style={'flex': '1', 'margin': '10px', 'minWidth': '45%'}),
                
                # Original Measurements over time
                html.Div([
                    html.H3("Distance Measurements", style={'textAlign': 'center'}),
                    html.P("Confronta i diversi metodi di calcolo della distanza in tempo reale, mostrando come si integrano tra loro.", 
                        style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '14px', 'paddingBottom': '10px'}),
                    dcc.Graph(
                        id='distance-plot',
                        style={'height': '60vh'},
                        config={'displayModeBar': False}
                    ),
                ], style={'flex': '1', 'margin': '10px', 'minWidth': '45%'})
            ], style={'display': 'flex', 'flexWrap': 'wrap'})
        ], style=tab_style, selected_style=tab_selected_style),
        
        # Tab 2: Anchor Analysis
        dcc.Tab(label="Anchor Analysis", children=[
            # RSSI vs Distance plots
            html.Div([
                html.H3("RSSI vs Distance per Anchor", style={'textAlign': 'center'}),
                html.P("Mostra la correlazione tra RSSI e distanza per l'anchor selezionato. Utile per valutare la qualità della stima della distanza basata sul segnale.", 
                    style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '14px', 'paddingBottom': '10px'}),
                html.Div([
                    # Dropdown per selezionare l'anchor
                    dcc.Dropdown(
                        id='anchor-dropdown',
                        options=[
                            {'label': f'Anchor {i}', 'value': f'anchor{i}'} 
                            for i in range(1, len(ANCHOR_POSITIONS) + 1)
                        ],
                        value='anchor1',  # Default value
                        style={'width': '200px', 'margin': '10px'}
                    ),
                ], style={'display': 'flex', 'justifyContent': 'center'}),
                
                # Grafico con asse Y doppio per RSSI e distanza
                dcc.Graph(
                    id='rssi-distance-plot',
                    style={'height': '40vh'},
                    config={'displayModeBar': False}
                ),
            ], style={'margin': '10px'}),
            
            # Signal Quality Indicators
            html.Div([
                html.H3("Signal Quality Indicators", style={'textAlign': 'center'}),
                html.P("Indicatori in tempo reale della qualità del segnale da ciascun anchor, utili per diagnosticare problemi di connettività e valutare l'affidabilità delle misurazioni.", 
                    style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '14px', 'paddingBottom': '10px'}),
                dcc.Graph(
                    id='signal-quality-plot',
                    style={'height': '40vh'},
                    config={'displayModeBar': False}
                ),
            ], style={'margin': '10px'}),
            
            # Anchor Coverage Map
            html.Div([
                html.H3("Anchor Coverage Map", style={'textAlign': 'center'}),
                html.P("Visualizza le aree meglio coperte dai vari anchor attraverso zone colorate. Utile per verificare la qualità del posizionamento degli anchor e identificare potenziali punti ciechi.", 
                    style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '14px', 'paddingBottom': '10px'}),
                dcc.Graph(
                    id='coverage-map-plot',
                    style={'height': '60vh'},
                    config={'displayModeBar': False}
                ),
            ], style={'margin': '10px'})
        ], style=tab_style, selected_style=tab_selected_style),
        
        # Tab 3: Movement Analysis
        dcc.Tab(label="Movement Analysis", children=[
            html.Div([
                # Position Heatmap
                html.Div([
                    html.H3("Position Heatmap", style={'textAlign': 'center'}),
                    html.P("Mostra la densità delle posizioni rilevate nell'area, evidenziando le zone più frequentate e i pattern di movimento.", 
                        style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '14px', 'paddingBottom': '10px'}),
                    dcc.Graph(
                        id='position-heatmap',
                        style={'height': '45vh'},
                        config={'displayModeBar': False}
                    ),
                ], style={'flex': '1', 'margin': '10px', 'minWidth': '45%'}),
                
                # Position Stability
                html.Div([
                    html.H3("Position Stability", style={'textAlign': 'center'}),
                    html.P("Visualizza la variazione della posizione nel tempo, quantificando la stabilità del sistema di tracciamento.", 
                        style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '14px', 'paddingBottom': '10px'}),
                    dcc.Graph(
                        id='position-stability-plot',
                        style={'height': '45vh'},
                        config={'displayModeBar': False}
                    ),
                ], style={'flex': '1', 'margin': '10px', 'minWidth': '45%'})
            ], style={'display': 'flex', 'flexWrap': 'wrap'}),
            
            html.Div([
                # Velocity Tracking
                html.Div([
                    html.H3("Velocity Over Time", style={'textAlign': 'center'}),
                    html.P("Mostra l'andamento della velocità nel tempo, utile per identificare momenti di movimento, fermo, accelerazione e decelerazione.", 
                        style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '14px', 'paddingBottom': '10px'}),
                    dcc.Graph(
                        id='velocity-plot',
                        style={'height': '45vh'},
                        config={'displayModeBar': False}
                    ),
                ], style={'flex': '1', 'margin': '10px', 'minWidth': '45%'}),
                
                # Activity Timeline
                html.Div([
                    html.H3("Activity Timeline", style={'textAlign': 'center'}),
                    html.P("Mostra i periodi di movimento e di riposo nel tempo, offrendo una panoramica dell'attività dell'utente.", 
                        style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '14px', 'paddingBottom': '10px'}),
                    dcc.Graph(
                        id='activity-timeline-plot',
                        style={'height': '45vh'},
                        config={'displayModeBar': False}
                    ),
                ], style={'flex': '1', 'margin': '10px', 'minWidth': '45%'})
            ], style={'display': 'flex', 'flexWrap': 'wrap'})
        ], style=tab_style, selected_style=tab_selected_style),
        
        # Tab 4: Distance Comparison
        dcc.Tab(label="Distance Analysis", children=[
            html.Div([
                html.H3("Distance Calculation Methods Comparison", style={'textAlign': 'center'}),
                html.P("Confronta i tre metodi di calcolo della distanza (basato sui passi, sulla posizione e fuso) mostrando l'evoluzione cumulativa nel tempo.", 
                    style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '14px', 'paddingBottom': '10px'}),
                dcc.Graph(
                    id='distance-comparison-plot',
                    style={'height': '60vh'},
                    config={'displayModeBar': False}
                ),
            ], style={'margin': '10px'})
        ], style=tab_style, selected_style=tab_selected_style)
    ]),
    
    # Update interval
    dcc.Interval(
        id='interval-component',
        interval=UPDATE_INTERVAL,
        n_intervals=0
    ),
    
    # Store current data state
    dcc.Store(id='data-timestamp', data=""),
    
    # Store processed data for multiple callbacks
    dcc.Store(id='processed-data', data=None)
])

# Callback to load and process data once
@app.callback(
    Output('processed-data', 'data'),
    [Input('interval-component', 'n_intervals'),
     Input('data-timestamp', 'data')]
)
def process_data(n_intervals, current_timestamp):
    csv_path = get_csv_path()
    if not csv_path or not os.path.exists(csv_path):
        return None
        
    try:
        csv_mtime = os.path.getmtime(csv_path)
        
        # If the file hasn't been modified since last check, return current state
        if str(csv_mtime) == current_timestamp and n_intervals > 0:
            return None
            
        # Read the CSV file
        df = pd.read_csv(csv_path, low_memory=False)
        
        if df.empty:
            return None
            
        # Convert timestamp to datetime and ensure it's sorted
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Convert string columns to numeric where needed
        numeric_columns = ['estimated_x', 'estimated_y', 'steps', 
                          'step_distance_m', 'position_distance_m', 'fused_distance_m',
                          'total_step_distance_m', 'total_position_distance_m', 'total_fused_distance_m']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert RSSI and distance columns for all anchors
        for anchor_id in ANCHOR_POSITIONS.keys():
            rssi_col = f'rssi_{anchor_id}'
            dist_col = f'distance_{anchor_id}'
            
            if rssi_col in df.columns:
                df[rssi_col] = pd.to_numeric(df[rssi_col], errors='coerce')
            if dist_col in df.columns:
                df[dist_col] = pd.to_numeric(df[dist_col], errors='coerce')
        
        # Calculate additional metrics
        
        # 1. Calculate velocity (m/s) with rolling window
        if len(df) > 2:
            # Create a rolling window of 3 points
            df['velocity'] = np.nan
            for i in range(2, len(df)):
                # Get the time difference in seconds
                time_diff = (df['timestamp'].iloc[i] - df['timestamp'].iloc[i-2]).total_seconds()
                if time_diff > 0:
                    # Calculate distance between points
                    if not pd.isna(df['estimated_x'].iloc[i]) and not pd.isna(df['estimated_y'].iloc[i]) and \
                       not pd.isna(df['estimated_x'].iloc[i-2]) and not pd.isna(df['estimated_y'].iloc[i-2]):
                        dx = df['estimated_x'].iloc[i] - df['estimated_x'].iloc[i-2]
                        dy = df['estimated_y'].iloc[i] - df['estimated_y'].iloc[i-2]
                        distance = np.sqrt(dx**2 + dy**2)
                        df.loc[df.index[i], 'velocity'] = distance / time_diff
        
        # 2. Calculate position stability (variation from moving average)
        if len(df) > 5:
            # Calculate rolling mean of position
            df['x_rolling_mean'] = df['estimated_x'].rolling(window=5, min_periods=1).mean()
            df['y_rolling_mean'] = df['estimated_y'].rolling(window=5, min_periods=1).mean()
            
            # Calculate distance from rolling mean (stability metric)
            df['position_variation'] = np.sqrt(
                (df['estimated_x'] - df['x_rolling_mean'])**2 + 
                (df['estimated_y'] - df['y_rolling_mean'])**2
            )
        
        # 3. Detect activity status (moving vs stationary)
        # Define a threshold for movement detection (adjust as needed)
        velocity_threshold = 0.1  # m/s
        
        # Create activity status column (1 = moving, 0 = stationary)
        df['is_moving'] = (df['velocity'] > velocity_threshold).astype(int)
        
        # Store file modification time
        data = {
            'dataframe': df.to_json(date_format='iso', orient='split'),
            'timestamp': str(csv_mtime)
        }
        
        return data
        
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

# Original callbacks for main metrics
@app.callback(
    [Output('total-distance', 'children'),
     Output('current-speed', 'children'),
     Output('total-steps', 'children'),
     Output('step-rate', 'children'),
     Output('data-timestamp', 'data')],
    [Input('processed-data', 'data'),
     Input('data-timestamp', 'data')]
)
def update_metrics(processed_data, current_timestamp):
    if not processed_data:
        return "0.00", "0.00", "0", "0.0", current_timestamp
    
    try:
        # Update timestamp
        new_timestamp = processed_data['timestamp']
        
        # Parse dataframe
        #df =ì pd.read_json(processed_data['dataframe'], orient='split')
        df = pd.read_json(StringIO(processed_data['dataframe']), orient='split')
        
        # Get the latest valid values
        total_fused_distance = 0
        latest_steps = 0
        
        # Find the last row with valid total_fused_distance_m
        fused_distance_series = df['total_fused_distance_m'].dropna()
        if not fused_distance_series.empty:
            total_fused_distance = fused_distance_series.iloc[-1]
            
        # Find the last row with valid steps
        steps_series = df['steps'].dropna()
        if not steps_series.empty:
            latest_steps = steps_series.iloc[-1]
        
        # Calculate speed (from processed velocity)
        speed = 0
        if 'velocity' in df.columns:
            velocity_series = df['velocity'].dropna()
            if not velocity_series.empty:
                # Get the average velocity over last 5 points
                speed = velocity_series.tail(5).mean()
        
        # Calculate step rate (steps per minute)
        step_rate = 0
        if len(df) > 1 and 'steps' in df.columns:
            # Get data from the last minute
            one_minute_ago = df['timestamp'].iloc[-1] - pd.Timedelta(minutes=1)
            minute_df = df[df['timestamp'] > one_minute_ago]
            
            if len(minute_df) > 1:
                step_values = minute_df['steps'].dropna()
                if len(step_values) > 1:
                    first_steps = step_values.iloc[0]
                    last_steps = step_values.iloc[-1]
                    step_diff = last_steps - first_steps
                    time_diff_minutes = (minute_df['timestamp'].iloc[-1] - minute_df['timestamp'].iloc[0]).total_seconds() / 60
                    
                    if time_diff_minutes > 0:
                        step_rate = step_diff / time_diff_minutes
        
        return f"{total_fused_distance:.2f}", f"{speed:.2f}", f"{latest_steps:.0f}", f"{step_rate:.1f}", new_timestamp
        
    except Exception as e:
        print(f"Error updating metrics: {e}")
        return "0.00", "0.00", "0", "0.0", current_timestamp

# Original callbacks for main plots
@app.callback(
    [Output('position-plot', 'figure'),
     Output('distance-plot', 'figure')],
    [Input('processed-data', 'data')]
)
def update_main_plots(processed_data):
    if not processed_data:
        return create_position_plot_with_anchors(), create_default_distance_plot()
    
    try:
        # Parse dataframe
        df = pd.read_json(StringIO(processed_data['dataframe']), orient='split')
        # Filter out rows with missing position data
        position_df = df.dropna(subset=['estimated_x', 'estimated_y'])
        
        # Create position plot
        if position_df.empty:
            position_plot = create_position_plot_with_anchors()
        else:
            path_x = position_df['estimated_x'].tolist()
            path_y = position_df['estimated_y'].tolist()
            
            # Ensure the current position point exists
            current_x = path_x[-1] if path_x else None
            current_y = path_y[-1] if path_y else None
            
            position_plot = {
                'data': [
                    # Plot anchors
                    go.Scatter(
                        x=[ANCHOR_POSITIONS[anchor][0] for anchor in ANCHOR_POSITIONS],
                        y=[ANCHOR_POSITIONS[anchor][1] for anchor in ANCHOR_POSITIONS],
                        text=[anchor for anchor in ANCHOR_POSITIONS],
                        mode='markers+text',
                        marker=dict(size=15, color='blue'),
                        textposition='top center',
                        name='Anchors'
                    ),
                    
                    # Plot wearable path - only if we have at least two points
                    go.Scatter(
                        x=path_x,
                        y=path_y,
                        mode='lines',
                        line=dict(width=2, color='lightgrey'),
                        name='Path'
                    ),
                ]
            }
            
            # Add current position as a separate trace if we have valid coordinates
            if current_x is not None and current_y is not None:
                position_plot['data'].append(
                    go.Scatter(
                        x=[current_x],
                        y=[current_y],
                        mode='markers+text',
                        marker=dict(size=20, color='red'),
                        text=['Current'],
                        textposition='top center',
                        name='Current Position'
                    )
                )
            
            position_plot['layout'] = go.Layout(
                xaxis=dict(
                    title='X Position (meters)',
                    range=[-1, 7],
                    gridcolor='lightgrey'
                ),
                yaxis=dict(
                    title='Y Position (meters)',
                    range=[-1, 7],
                    gridcolor='lightgrey'
                ),
                margin=dict(l=40, r=40, t=10, b=40),
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=True,
                legend=dict(x=0, y=1),
                hovermode='closest'
            )
        
        # Create distance over time plot
        # Limit to last MAX_POINTS points for performance
        recent_df = df.tail(MAX_POINTS) if len(df) > MAX_POINTS else df
        
        if recent_df.empty or 'step_distance_m' not in recent_df.columns:
            distance_plot = create_default_distance_plot()
        else:
            # Make sure we have data for each series
            data_series = []
            
            # Only add series that have valid data
            if not recent_df['step_distance_m'].dropna().empty:
                data_series.append(
                    go.Scatter(
                        x=recent_df['timestamp'],
                        y=recent_df['step_distance_m'],
                        mode='lines',
                        name='Step Distance',
                        line=dict(color='green', width=1)
                    )
                )
                
            if not recent_df['position_distance_m'].dropna().empty:
                data_series.append(
                    go.Scatter(
                        x=recent_df['timestamp'],
                        y=recent_df['position_distance_m'],
                        mode='lines',
                        name='Position Distance',
                        line=dict(color='blue', width=1)
                    )
                )
                
            if not recent_df['fused_distance_m'].dropna().empty:
                data_series.append(
                    go.Scatter(
                        x=recent_df['timestamp'],
                        y=recent_df['fused_distance_m'],
                        mode='lines+markers',
                        name='Fused Distance',
                        line=dict(color='red', width=2),
                        marker=dict(size=5)
                    )
                )
            
            distance_plot = {
                'data': data_series,
                'layout': go.Layout(
                    xaxis=dict(
                        title='Time',
                        gridcolor='lightgrey'
                    ),
                    yaxis=dict(
                        title='Distance (meters)',
                        gridcolor='lightgrey'
                    ),
                    margin=dict(l=40, r=40, t=10, b=40),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    showlegend=True,
                    legend=dict(x=0, y=1),
                    hovermode='closest'
                )
            }
        
        return position_plot, distance_plot
        
    except Exception as e:
        print(f"Error updating main plots: {e}")
        return create_position_plot_with_anchors(), create_default_distance_plot()

# Callback for RSSI vs Distance plot
@app.callback(
    Output('rssi-distance-plot', 'figure'),
    [Input('processed-data', 'data'),
     Input('anchor-dropdown', 'value')]
)
def update_rssi_distance_plot(processed_data, selected_anchor):
    if not processed_data or not selected_anchor:
        return create_default_rssi_distance_plot(selected_anchor)
    
    try:
        # Parse dataframe
        df = pd.read_json(StringIO(processed_data['dataframe']), orient='split')
        # Get RSSI and distance columns for the selected anchor
        rssi_col = f'rssi_{selected_anchor}'
        dist_col = f'distance_{selected_anchor}'
        
        # Check if we have the required columns
        if rssi_col in df.columns and dist_col in df.columns:
            # Drop rows with missing data
            valid_data = df.dropna(subset=[rssi_col, dist_col])
            
            if not valid_data.empty:
                # Create subplot with shared x-axis and two y-axes
                fig = sp.make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add RSSI trace (primary y-axis)
                fig.add_trace(
                    go.Scatter(
                        x=valid_data['timestamp'],
                        y=valid_data[rssi_col],
                        name='RSSI',
                        line=dict(color='blue', width=2)
                    ),
                    secondary_y=False
                )
                
                # Add distance trace (secondary y-axis)
                fig.add_trace(
                    go.Scatter(
                        x=valid_data['timestamp'],
                        y=valid_data[dist_col],
                        name='Distance',
                        line=dict(color='red', width=2)
                    ),
                    secondary_y=True
                )
                
                # Set axis titles
                fig.update_layout(
                    title=f'RSSI vs Distance for {selected_anchor}',
                    xaxis=dict(title='Time', gridcolor='lightgrey'),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=50, r=50, t=30, b=40)
                )
                
                # Set y-axes titles
                fig.update_yaxes(title_text="RSSI (dBm)", secondary_y=False)
                fig.update_yaxes(title_text="Distance (meters)", secondary_y=True)
                
                return fig
    
        return create_default_rssi_distance_plot(selected_anchor)
        
    except Exception as e:
        print(f"Error updating RSSI-Distance plot: {e}")
        return create_default_rssi_distance_plot(selected_anchor)

# Callback for Signal Quality Indicators
@app.callback(
    Output('signal-quality-plot', 'figure'),
    [Input('processed-data', 'data')]
)
def update_signal_quality_plot(processed_data):
    if not processed_data:
        return create_default_signal_quality_plot()
    
    try:
        # Parse dataframe
        df = pd.read_json(StringIO(processed_data['dataframe']), orient='split')
        # Create signal quality figure
        fig = go.Figure()
        
        # Add a trace for each anchor
        for anchor_id in ANCHOR_POSITIONS.keys():
            rssi_col = f'rssi_{anchor_id}'
            
            if rssi_col in df.columns:
                # Drop rows with missing data
                valid_data = df.dropna(subset=[rssi_col])
                
                if not valid_data.empty:
                    # Get last few points for recent signal quality
                    recent_data = valid_data.tail(MAX_POINTS)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=recent_data['timestamp'],
                            y=recent_data[rssi_col],
                            name=f'{anchor_id} RSSI',
                            line=dict(width=2)
                        )
                    )
        
        # Update layout
        fig.update_layout(
            xaxis=dict(title='Time', gridcolor='lightgrey'),
            yaxis=dict(title='RSSI (dBm)', gridcolor='lightgrey'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=50, t=30, b=40)
        )
        
        return fig
    
    except Exception as e:
        print(f"Error updating signal quality plot: {e}")
        return create_default_signal_quality_plot()

# Callback for Coverage Map
@app.callback(
    Output('coverage-map-plot', 'figure'),
    [Input('processed-data', 'data')]
)
def update_coverage_map_plot(processed_data):
    if not processed_data:
        return create_default_coverage_map_plot()
    
    try:
        # Parse dataframe
        df = pd.read_json(StringIO(processed_data['dataframe']), orient='split')
        # Create a heatmap grid
        x_grid = np.linspace(-0.5, 5.5, 25)
        y_grid = np.linspace(-0.5, 5.5, 25)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Create a figure
        fig = go.Figure()
        
        # Add anchor markers
        fig.add_trace(
            go.Scatter(
                x=[ANCHOR_POSITIONS[anchor][0] for anchor in ANCHOR_POSITIONS],
                y=[ANCHOR_POSITIONS[anchor][1] for anchor in ANCHOR_POSITIONS],
                text=[anchor for anchor in ANCHOR_POSITIONS],
                mode='markers+text',
                marker=dict(size=15, color='black'),
                textposition='top center',
                name='Anchors'
            )
        )
        
        # Add coverage contours for each anchor
        for i, anchor_id in enumerate(ANCHOR_POSITIONS.keys()):
            # Create a coverage map based on distance from anchor
            Z = np.sqrt((X - ANCHOR_POSITIONS[anchor_id][0])**2 + (Y - ANCHOR_POSITIONS[anchor_id][1])**2)
            
            # Invert the values so that closer to anchor is higher value
            Z = 1 / (Z + 0.1)  # Adding 0.1 to avoid division by zero
            
            # Add contour
            fig.add_trace(
                go.Contour(
                    z=Z,
                    x=x_grid,
                    y=y_grid,
                    colorscale=f'Blues',
                    opacity=0.5,
                    showscale=False,
                    name=anchor_id,
                    contours=dict(
                        showlabels=False,
                        coloring='heatmap'
                    )
                )
            )
        
        # Update layout
        fig.update_layout(
            xaxis=dict(
                title='X Position (meters)',
                range=[-1, 7],
                gridcolor='lightgrey'
            ),
            yaxis=dict(
                title='Y Position (meters)',
                range=[-1, 7],
                gridcolor='lightgrey'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=30, b=40)
        )
        
        return fig
        
    except Exception as e:
        print(f"Error updating coverage map plot: {e}")
        return create_default_coverage_map_plot()

# Callback for Position Heatmap
@app.callback(
    Output('position-heatmap', 'figure'),
    [Input('processed-data', 'data')]
)
def update_position_heatmap(processed_data):
    if not processed_data:
        return create_default_position_heatmap()
    
    try:
        # Parse dataframe
        df = pd.read_json(StringIO(processed_data['dataframe']), orient='split')
        # Filter out rows with missing position data
        position_df = df.dropna(subset=['estimated_x', 'estimated_y'])
        
        if position_df.empty:
            return create_default_position_heatmap()
        
        # Create a 2D histogram heatmap
        fig = ff.create_2d_density(
            position_df['estimated_x'],
            position_df['estimated_y'],
            colorscale='Viridis',
            hist_color='rgb(255, 237, 222)',
            point_size=3
        )
        
        # Add anchors
        fig.add_trace(
            go.Scatter(
                x=[ANCHOR_POSITIONS[anchor][0] for anchor in ANCHOR_POSITIONS],
                y=[ANCHOR_POSITIONS[anchor][1] for anchor in ANCHOR_POSITIONS],
                text=[anchor for anchor in ANCHOR_POSITIONS],
                mode='markers+text',
                marker=dict(size=15, color='red'),
                textposition='top center',
                name='Anchors'
            )
        )
        
        # Update layout
        fig.update_layout(
            xaxis=dict(
                title='X Position (meters)',
                range=[-1, 7],
                gridcolor='lightgrey'
            ),
            yaxis=dict(
                title='Y Position (meters)',
                range=[-1, 7],
                gridcolor='lightgrey'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=30, b=40)
        )
        
        return fig
        
    except Exception as e:
        print(f"Error updating position heatmap: {e}")
        return create_default_position_heatmap()

# Callback for Position Stability
@app.callback(
    Output('position-stability-plot', 'figure'),
    [Input('processed-data', 'data')]
)
def update_position_stability_plot(processed_data):
    if not processed_data:
        return create_default_stability_plot()
    
    try:
        # Parse dataframe
        df = pd.read_json(StringIO(processed_data['dataframe']), orient='split')
        # Check if we have position variation data
        if 'position_variation' not in df.columns or df['position_variation'].dropna().empty:
            return create_default_stability_plot()
        
        # Limit to recent data points
        recent_df = df.tail(MAX_POINTS) if len(df) > MAX_POINTS else df
        
        # Create stability plot
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=recent_df['timestamp'],
                y=recent_df['position_variation'],
                mode='lines',
                name='Position Variation',
                line=dict(color='purple', width=2)
            )
        )
        
        # Add a rolling average line
        if len(recent_df) > 5:
            fig.add_trace(
                go.Scatter(
                    x=recent_df['timestamp'],
                    y=recent_df['position_variation'].rolling(window=5).mean(),
                    mode='lines',
                    name='Rolling Average (5 points)',
                    line=dict(color='red', width=2, dash='dash')
                )
            )
        
        # Update layout
        fig.update_layout(
            xaxis=dict(title='Time', gridcolor='lightgrey'),
            yaxis=dict(title='Position Variation (meters)', gridcolor='lightgrey'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=50, t=30, b=40)
        )
        
        return fig
        
    except Exception as e:
        print(f"Error updating position stability plot: {e}")
        return create_default_stability_plot()

# Callback for Velocity Plot
@app.callback(
    Output('velocity-plot', 'figure'),
    [Input('processed-data', 'data')]
)
def update_velocity_plot(processed_data):
    if not processed_data:
        return create_default_velocity_plot()
    
    try:
        # Parse dataframe
        df = pd.read_json(StringIO(processed_data['dataframe']), orient='split')
        # Check if we have velocity data
        if 'velocity' not in df.columns or df['velocity'].dropna().empty:
            return create_default_velocity_plot()
        
        # Limit to recent data points
        recent_df = df.tail(MAX_POINTS) if len(df) > MAX_POINTS else df
        
        # Create velocity plot
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=recent_df['timestamp'],
                y=recent_df['velocity'],
                mode='lines',
                name='Velocity',
                line=dict(color='orange', width=2)
            )
        )
        
        # Update layout
        fig.update_layout(
            xaxis=dict(title='Time', gridcolor='lightgrey'),
            yaxis=dict(title='Velocity (m/s)', gridcolor='lightgrey'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=50, t=30, b=40)
        )
        
        return fig
        
    except Exception as e:
        print(f"Error updating velocity plot: {e}")
        return create_default_velocity_plot()

# Callback for Activity Timeline
@app.callback(
    Output('activity-timeline-plot', 'figure'),
    [Input('processed-data', 'data')]
)
def update_activity_timeline_plot(processed_data):
    if not processed_data:
        return create_default_activity_timeline()
    
    try:
        # Parse dataframe
        df = pd.read_json(StringIO(processed_data['dataframe']), orient='split')
        # Check if we have activity data
        if 'is_moving' not in df.columns or df['is_moving'].dropna().empty:
            return create_default_activity_timeline()
        
        # Create activity timeline
        fig = go.Figure()
        
        # Add activity status as colored bars
        # First, find segments of continuous activity
        activity_changes = df['is_moving'].diff().fillna(0) != 0
        change_indices = df.index[activity_changes].tolist()
        
        # Add start and end indices
        if 0 not in change_indices:
            change_indices = [0] + change_indices
        if len(df) - 1 not in change_indices:
            change_indices = change_indices + [len(df) - 1]
        
        # Sort indices
        change_indices = sorted(change_indices)
        
        # Create segments
        for i in range(len(change_indices) - 1):
            start_idx = change_indices[i]
            end_idx = change_indices[i + 1]
            
            # Get activity status for this segment
            is_moving = df['is_moving'].iloc[start_idx]
            
            # Set color based on activity status
            color = 'rgba(0, 255, 0, 0.3)' if is_moving else 'rgba(255, 0, 0, 0.3)'
            label = 'Moving' if is_moving else 'Stationary'
            
            # Add a rectangle for this segment
            fig.add_shape(
                type="rect",
                x0=df['timestamp'].iloc[start_idx],
                x1=df['timestamp'].iloc[end_idx],
                y0=0,
                y1=1,
                fillcolor=color,
                opacity=0.5,
                layer="below",
                line_width=0,
            )
        
        # Add the velocity line for reference
        if 'velocity' in df.columns and not df['velocity'].dropna().empty:
            # Scale velocity to fit in the same plot
            max_vel = df['velocity'].max()
            if max_vel > 0:
                scaled_velocity = df['velocity'] / max_vel
                
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=scaled_velocity,
                        mode='lines',
                        name='Velocity (scaled)',
                        line=dict(color='blue', width=2)
                    )
                )
        
        # Update layout
        fig.update_layout(
            xaxis=dict(title='Time', gridcolor='lightgrey'),
            yaxis=dict(
                title='Activity',
                tickvals=[0, 0.5, 1],
                ticktext=['', 'Activity Status', ''],
                gridcolor='lightgrey',
                range=[0, 1]
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=50, t=30, b=40)
        )
        
        # Add legend for activity status
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=15, color='rgba(0, 255, 0, 0.3)'),
            name='Moving'
        ))
        
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=15, color='rgba(255, 0, 0, 0.3)'),
            name='Stationary'
        ))
        
        return fig
        
    except Exception as e:
        print(f"Error updating activity timeline plot: {e}")
        return create_default_activity_timeline()

# Callback for Distance Comparison
@app.callback(
    Output('distance-comparison-plot', 'figure'),
    [Input('processed-data', 'data')]
)
def update_distance_comparison_plot(processed_data):
    if not processed_data:
        return create_default_distance_comparison_plot()
    
    try:
        # Parse dataframe
        df = pd.read_json(StringIO(processed_data['dataframe']), orient='split')
        # Check if we have the necessary columns
        required_cols = ['total_step_distance_m', 'total_position_distance_m', 'total_fused_distance_m']
        if not all(col in df.columns for col in required_cols):
            return create_default_distance_comparison_plot()
        
        # Create figure
        fig = go.Figure()
        
        # Add traces for each distance calculation method
        for col, name, color in zip(
            required_cols,
            ['Step Distance', 'Position Distance', 'Fused Distance'],
            ['green', 'blue', 'red']
        ):
            # Drop rows with missing data
            valid_data = df.dropna(subset=[col])
            
            if not valid_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=valid_data['timestamp'],
                        y=valid_data[col],
                        mode='lines',
                        name=name,
                        line=dict(color=color, width=2)
                    )
                )
        
        # Update layout
        fig.update_layout(
            xaxis=dict(title='Time', gridcolor='lightgrey'),
            yaxis=dict(title='Distance (meters)', gridcolor='lightgrey'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=50, t=30, b=40)
        )
        
        return fig
        
    except Exception as e:
        print(f"Error updating distance comparison plot: {e}")
        return create_default_distance_comparison_plot()


#______________________added by me_______________________

# Callback per aggiornare il display
@app.callback(
    Output('timer-display', 'children'),
    Output('timer-display', 'style'),
    Input('timer-interval', 'n_intervals'),
    State('start-time', 'data')
)

def update_timer(n, start_time_iso):
    if not start_time_iso:
        return "Timer not started", {'color': 'black'}

    start_time = datetime.fromisoformat(start_time_iso)
    elapsed = (datetime.now() - start_time).total_seconds()

    mins_elapsed = int(elapsed // 60)
    secs_elapsed = int(elapsed % 60)

    if elapsed < 360:
        mins_remaining = int((360 - elapsed) // 60)
        secs_remaining = int((360 - elapsed) % 60)
        text = f"Elapsed: {mins_elapsed:02}:{secs_elapsed:02} | Remaining: {mins_remaining:02}:{secs_remaining:02}"
        style = {'color': 'green'}
    else:
        text = f"Elapsed: {mins_elapsed:02}:{secs_elapsed:02} | 6 minutes over! ⏰"
        style = {'color': 'red', 'fontWeight': 'bold'}

    return text, style




# callback per entrambi i buttons, start e reset button
@app.callback(
    Output('start-time', 'data'),
    Output('timer-interval', 'disabled'),
    Input('start-button', 'n_clicks'),
    Input('reset-button', 'n_clicks')
)
def manage_timer(start_clicks, reset_clicks):
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]
    
    if triggered_id == 'start-button':
        return datetime.now().isoformat(), False
    elif triggered_id == 'reset-button':
        return None, True
    return dash.no_update, dash.no_update

# callback to write the json file
@app.callback(
    Output('patient-confirmation-msg', 'children'),
    Input('confirm-patient-btn', 'n_clicks'),
    State('patient-name', 'value'),
    State('patient-surname', 'value'),
    State('patient-height', 'value'),
    State('patient-gender', 'value'),
    State('patient-age', 'value')
)
def save_patient_info(n_clicks, name, surname, height, gender, age):
    # Add print statements to debug
    print(f"Save callback triggered: clicks={n_clicks}, name={name}, surname={surname}")
    
    if n_clicks is None or n_clicks == 0:
        return "Waiting for confirmation..."

    if not all([name, surname, height, gender, age]):
        return "Please fill all fields."

    patient_id = f"{name.strip().lower()}_{surname.strip().lower()}".replace(" ", "_")
    data = {
        "id": patient_id,
        "height": float(height),
        "gender": gender.lower(),
        "age": int(age)
    }

    try:
        # Use absolute path to ensure we know where the file is saved
        file_path = os.path.join(os.getcwd(), "current_patient.json")
        print(f"Attempting to save to: {file_path}")
        
        with open(file_path, "w") as f:
            json.dump(data, f)
        
        # Verify the file was created
        if os.path.exists(file_path):
            print(f"File successfully created at {file_path}")
            return f"Patient data saved: {patient_id} at {file_path}"
        else:
            return f"File was not created at {file_path}"
    except Exception as e:
        print(f"Error saving file: {str(e)}")
        return f"Error saving patient data: {str(e)}"

# Modified callback to validate data but not disable the start button
@app.callback(
    Output('patient-data-valid', 'data'),
    Output('data-warning-msg', 'children'),
    Input('confirm-patient-btn', 'n_clicks'),
    State('patient-name', 'value'),
    State('patient-surname', 'value'),
    State('patient-height', 'value'),
    State('patient-gender', 'value'),
    State('patient-age', 'value'),
    prevent_initial_call=True
)
def validate_patient_data(n_clicks, name, surname, height, gender, age):
    if n_clicks == 0:
        return False, ""

    if not all([name, surname, height, gender, age]):
        return False, "Please fill in all patient data before starting the test."

    try:
        height_val = float(height)
        age_val = int(age)
        if height_val <= 0 or age_val <= 0:
            return False, "Height and age must be positive values."
    except ValueError:
        return False, "Height must be a number and age must be an integer."

    return True, ""  # dati validi

# Callback per salvare il paziente e lanciare il backend
# Variabile globale per mantenere il processo server (solo in questa sessione)
server_process = None

@app.callback(
    Output("test-status-msg", "children", allow_duplicate=True),
    Output("check-process", "disabled", allow_duplicate=True),
    Output("test-running", "data", allow_duplicate=True),
    Input("start-test-button", "n_clicks"),
    State("patient-name", "value"),
    State("patient-surname", "value"),
    State("patient-height", "value"),
    State("patient-gender", "value"),
    State("patient-age", "value"),
    prevent_initial_call=True
)
def start_test_process(n_clicks, name, surname, height, gender, age):
    global server_process

    if not all([name, surname, height, gender, age]):
        return "Please fill all patient fields.", True, False

    patient_id = f"{name.strip().lower()}_{surname.strip().lower()}".replace(" ", "_")
    patient_data = {
        "id": patient_id,
        "height": float(height),
        "gender": gender.lower(),
        "age": int(age)
    }

    try:
        with open("current_patient.json", "w") as f:
            json.dump(patient_data, f)

        # Avvia il server come processo separato
        server_process = subprocess.Popen(["python", "enhanced-distance-tracking_v4.py"])
        return f"Test started for {patient_id}.", False, True
    except Exception as e:
        return f"Error: {e}", True, False

# Callback per controllare se il processo è finito
@app.callback(
    Output("test-status-msg", "children", allow_duplicate=True),
    Output("check-process", "disabled", allow_duplicate=True),
    Output("test-running", "data", allow_duplicate=True),
    Input("check-process", "n_intervals"),
    prevent_initial_call="initial_duplicate"
)
def check_server_status(n):
    global server_process
    if server_process and server_process.poll() is not None:
        server_process = None
        return "Test completed. You may now start a new one.", True, False
    return dash.no_update, False, True




# Helper functions for creating default plots when data is not available
def create_position_plot_with_anchors():
    return {
        'data': [
            # Plot anchors
            go.Scatter(
                x=[ANCHOR_POSITIONS[anchor][0] for anchor in ANCHOR_POSITIONS],
                y=[ANCHOR_POSITIONS[anchor][1] for anchor in ANCHOR_POSITIONS],
                text=[anchor for anchor in ANCHOR_POSITIONS],
                mode='markers+text',
                marker=dict(size=15, color='blue'),
                textposition='top center',
                name='Anchors'
            )
        ],
        'layout': go.Layout(
            xaxis=dict(
                title='X Position (meters)',
                range=[-1, 7],
                gridcolor='lightgrey'
            ),
            yaxis=dict(
                title='Y Position (meters)',
                range=[-1, 7],
                gridcolor='lightgrey'
            ),
            margin=dict(l=40, r=40, t=10, b=40),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(x=0, y=1),
            hovermode='closest'
        )
    }

def create_default_distance_plot():
    return {
        'data': [],
        'layout': go.Layout(
            xaxis=dict(
                title='Time',
                gridcolor='lightgrey'
            ),
            yaxis=dict(
                title='Distance (meters)',
                gridcolor='lightgrey'
            ),
            margin=dict(l=40, r=40, t=10, b=40),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(x=0, y=1),
            hovermode='closest',
            annotations=[
                dict(
                    text="Waiting for data...",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    font=dict(size=16)
                )
            ]
        )
    }

def create_default_rssi_distance_plot(selected_anchor):
    return {
        'data': [],
        'layout': go.Layout(
            title=f'RSSI vs Distance for {selected_anchor}',
            xaxis=dict(title='Time', gridcolor='lightgrey'),
            yaxis=dict(title='Value', gridcolor='lightgrey'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=50, b=40),
            annotations=[
                dict(
                    text="Waiting for data...",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    font=dict(size=16)
                )
            ]
        )
    }

def create_default_signal_quality_plot():
    return {
        'data': [],
        'layout': go.Layout(
            xaxis=dict(title='Time', gridcolor='lightgrey'),
            yaxis=dict(title='RSSI (dBm)', gridcolor='lightgrey'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=30, b=40),
            annotations=[
                dict(
                    text="Waiting for data...",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    font=dict(size=16)
                )
            ]
        )
    }

def create_default_coverage_map_plot():
    return {
        'data': [
            # Plot anchors
            go.Scatter(
                x=[ANCHOR_POSITIONS[anchor][0] for anchor in ANCHOR_POSITIONS],
                y=[ANCHOR_POSITIONS[anchor][1] for anchor in ANCHOR_POSITIONS],
                text=[anchor for anchor in ANCHOR_POSITIONS],
                mode='markers+text',
                marker=dict(size=15, color='black'),
                textposition='top center',
                name='Anchors'
            )
        ],
        'layout': go.Layout(
            xaxis=dict(
                title='X Position (meters)',
                range=[-1, 7],
                gridcolor='lightgrey'
            ),
            yaxis=dict(
                title='Y Position (meters)',
                range=[-1, 7],
                gridcolor='lightgrey'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=30, b=40),
            annotations=[
                dict(
                    text="Waiting for data...",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    font=dict(size=16)
                )
            ]
        )
    }

def create_default_position_heatmap():
    return {
        'data': [
            # Plot anchors
            go.Scatter(
                x=[ANCHOR_POSITIONS[anchor][0] for anchor in ANCHOR_POSITIONS],
                y=[ANCHOR_POSITIONS[anchor][1] for anchor in ANCHOR_POSITIONS],
                text=[anchor for anchor in ANCHOR_POSITIONS],
                mode='markers+text',
                marker=dict(size=15, color='red'),
                textposition='top center',
                name='Anchors'
            )
        ],
        'layout': go.Layout(
            xaxis=dict(
                title='X Position (meters)',
                range=[-1, 7],
                gridcolor='lightgrey'
            ),
            yaxis=dict(
                title='Y Position (meters)',
                range=[-1, 7],
                gridcolor='lightgrey'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=30, b=40),
            annotations=[
                dict(
                    text="Waiting for data...",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    font=dict(size=16)
                )
            ]
        )
    }

def create_default_stability_plot():
    return {
        'data': [],
        'layout': go.Layout(
            xaxis=dict(title='Time', gridcolor='lightgrey'),
            yaxis=dict(title='Position Variation (meters)', gridcolor='lightgrey'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=30, b=40),
            annotations=[
                dict(
                    text="Waiting for data...",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    font=dict(size=16)
                )
            ]
        )
    }

def create_default_velocity_plot():
    return {
        'data': [],
        'layout': go.Layout(
            xaxis=dict(title='Time', gridcolor='lightgrey'),
            yaxis=dict(title='Velocity (m/s)', gridcolor='lightgrey'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=30, b=40),
            annotations=[
                dict(
                    text="Waiting for data...",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    font=dict(size=16)
                )
            ]
        )
    }

def create_default_activity_timeline():
    return {
        'data': [],
        'layout': go.Layout(
            xaxis=dict(title='Time', gridcolor='lightgrey'),
            yaxis=dict(
                title='Activity',
                tickvals=[0, 0.5, 1],
                ticktext=['', 'Activity Status', ''],
                gridcolor='lightgrey',
                range=[0, 1]
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=30, b=40),
            annotations=[
                dict(
                    text="Waiting for data...",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    font=dict(size=16)
                )
            ]
        )
    }

def create_default_distance_comparison_plot():
    return {
        'data': [],
        'layout': go.Layout(
            xaxis=dict(title='Time', gridcolor='lightgrey'),
            yaxis=dict(title='Distance (meters)', gridcolor='lightgrey'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=30, b=40),
            annotations=[
                dict(
                    text="Waiting for data...",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    font=dict(size=16)
                )
            ]
        )
    }

# Start the server
if __name__ == '__main__':
    print("Starting Wearable Positioning System Dashboard")
    print("Dashboard will connect to latest data from the server")
    app.run(host='0.0.0.0', debug=True, port=8050)