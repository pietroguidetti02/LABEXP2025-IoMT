#!/usr/bin/env python3
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import os
import time
import math
import threading

# Configuration
UPDATE_INTERVAL = 1000  # Update interval in milliseconds
MAX_POINTS = 100  # Maximum number of points to display in time series

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

# Initialize the Dash app
app = dash.Dash(__name__, update_title=None)
app.title = "Wearable Positioning System"

# Define the layout
app.layout = html.Div([
    html.H1("Wearable Positioning and Distance Tracking System", 
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
    
    html.Div([
        # Statistics cards
        html.Div([
            html.Div([
                html.H4("Total Distance", style={'textAlign': 'center', 'margin': '5px'}),
                html.Div(id='total-distance', style={'textAlign': 'center', 'fontSize': '24px', 'fontWeight': 'bold'}),
                html.Div("meters", style={'textAlign': 'center', 'color': '#7f8c8d'})
            ], style={'flex': '1', 'backgroundColor': '#ecf0f1', 'borderRadius': '10px', 'padding': '15px', 'margin': '5px'}),
            
            html.Div([
                html.H4("Current Speed", style={'textAlign': 'center', 'margin': '5px'}),
                html.Div(id='current-speed', style={'textAlign': 'center', 'fontSize': '24px', 'fontWeight': 'bold'}),
                html.Div("m/s", style={'textAlign': 'center', 'color': '#7f8c8d'})
            ], style={'flex': '1', 'backgroundColor': '#ecf0f1', 'borderRadius': '10px', 'padding': '15px', 'margin': '5px'}),
            
            html.Div([
                html.H4("Total Steps", style={'textAlign': 'center', 'margin': '5px'}),
                html.Div(id='total-steps', style={'textAlign': 'center', 'fontSize': '24px', 'fontWeight': 'bold'}),
                html.Div("steps", style={'textAlign': 'center', 'color': '#7f8c8d'})
            ], style={'flex': '1', 'backgroundColor': '#ecf0f1', 'borderRadius': '10px', 'padding': '15px', 'margin': '5px'}),
            
            html.Div([
                html.H4("Step Rate", style={'textAlign': 'center', 'margin': '5px'}),
                html.Div(id='step-rate', style={'textAlign': 'center', 'fontSize': '24px', 'fontWeight': 'bold'}),
                html.Div("steps/min", style={'textAlign': 'center', 'color': '#7f8c8d'})
            ], style={'flex': '1', 'backgroundColor': '#ecf0f1', 'borderRadius': '10px', 'padding': '15px', 'margin': '5px'}),
        ], style={'display': 'flex', 'justifyContent': 'space-around', 'margin': '10px 0'}),
        
        # Main visualizations
        html.Div([
            # 2D Position plot
            html.Div([
                html.H3("Real-time Position", style={'textAlign': 'center'}),
                dcc.Graph(
                    id='position-plot',
                    style={'height': '60vh'},
                    config={'displayModeBar': False}
                ),
            ], style={'flex': '1', 'margin': '10px'}),
            
            # Measurements over time
            html.Div([
                html.H3("Distance Measurements", style={'textAlign': 'center'}),
                dcc.Graph(
                    id='distance-plot',
                    style={'height': '60vh'},
                    config={'displayModeBar': False}
                ),
            ], style={'flex': '1', 'margin': '10px'})
        ], style={'display': 'flex', 'flexWrap': 'wrap'}),
        
        # Update interval
        dcc.Interval(
            id='interval-component',
            interval=UPDATE_INTERVAL,
            n_intervals=0
        )
    ])
])

# Callback to update all components
@app.callback(
    [Output('position-plot', 'figure'),
     Output('distance-plot', 'figure'),
     Output('total-distance', 'children'),
     Output('current-speed', 'children'),
     Output('total-steps', 'children'),
     Output('step-rate', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_metrics(n_intervals):
    # Get data
    csv_path = get_csv_path()
    if csv_path and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter out rows with missing position data
            position_df = df.dropna(subset=['estimated_x', 'estimated_y'])
            
            # Get the latest values
            total_fused_distance = df['total_fused_distance_m'].iloc[-1] if not df.empty and 'total_fused_distance_m' in df else 0
            latest_steps = df['steps'].iloc[-1] if not df.empty and 'steps' in df else 0
            
            # Calculate speed (moving average over last 5 seconds)
            speed = 0
            if len(df) > 1 and 'timestamp' in df and 'fused_distance_m' in df:
                recent_df = df.tail(5)  # Last 5 measurements
                if len(recent_df) > 1:
                    time_diff = (recent_df['timestamp'].iloc[-1] - recent_df['timestamp'].iloc[0]).total_seconds()
                    if time_diff > 0:
                        distance_diff = recent_df['fused_distance_m'].sum()
                        speed = distance_diff / time_diff
            
            # Calculate step rate (steps per minute)
            step_rate = 0
            if len(df) > 1 and 'timestamp' in df and 'steps' in df:
                # Get data from the last minute
                one_minute_ago = df['timestamp'].iloc[-1] - pd.Timedelta(minutes=1)
                minute_df = df[df['timestamp'] > one_minute_ago]
                
                if len(minute_df) > 1:
                    first_steps = minute_df['steps'].iloc[0]
                    last_steps = minute_df['steps'].iloc[-1]
                    step_diff = last_steps - first_steps if not pd.isna(first_steps) and not pd.isna(last_steps) else 0
                    time_diff_minutes = (minute_df['timestamp'].iloc[-1] - minute_df['timestamp'].iloc[0]).total_seconds() / 60
                    
                    if time_diff_minutes > 0:
                        step_rate = step_diff / time_diff_minutes
            
            # Create position plot
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
                    
                    # Plot wearable path
                    go.Scatter(
                        x=position_df['estimated_x'],
                        y=position_df['estimated_y'],
                        mode='lines',
                        line=dict(width=2, color='lightgrey'),
                        name='Path'
                    ),
                    
                    # Plot current position
                    go.Scatter(
                        x=[position_df['estimated_x'].iloc[-1]] if len(position_df) > 0 else [],
                        y=[position_df['estimated_y'].iloc[-1]] if len(position_df) > 0 else [],
                        mode='markers+text',
                        marker=dict(size=20, color='red'),
                        text=['Current'],
                        textposition='top center',
                        name='Current Position'
                    ),
                ],
                'layout': go.Layout(
                    xaxis=dict(
                        title='X Position (meters)',
                        range=[-1, 11],
                        gridcolor='lightgrey'
                    ),
                    yaxis=dict(
                        title='Y Position (meters)',
                        range=[-1, 11],
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
            
            # Create distance over time plot
            # Limit to last MAX_POINTS points for performance
            recent_df = df.tail(MAX_POINTS) if len(df) > MAX_POINTS else df
            
            distance_plot = {
                'data': [
                    # Plot step-based distance
                    go.Scatter(
                        x=recent_df['timestamp'],
                        y=recent_df['step_distance_m'],
                        mode='lines',
                        name='Step Distance',
                        line=dict(color='green', width=1)
                    ),
                    
                    # Plot position-based distance
                    go.Scatter(
                        x=recent_df['timestamp'],
                        y=recent_df['position_distance_m'],
                        mode='lines',
                        name='Position Distance',
                        line=dict(color='blue', width=1)
                    ),
                    
                    # Plot fused distance
                    go.Scatter(
                        x=recent_df['timestamp'],
                        y=recent_df['fused_distance_m'],
                        mode='lines+markers',
                        name='Fused Distance',
                        line=dict(color='red', width=2),
                        marker=dict(size=5)
                    ),
                ],
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
            
            return position_plot, distance_plot, f"{total_fused_distance:.2f}", f"{speed:.2f}", f"{latest_steps}", f"{step_rate:.1f}"
            
        except Exception as e:
            print(f"Error updating plots: {e}")
    
    # Default empty plots if data not available
    default_position_plot = {
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
                range=[-1, 11],
                gridcolor='lightgrey'
            ),
            yaxis=dict(
                title='Y Position (meters)',
                range=[-1, 11],
                gridcolor='lightgrey'
            ),
            margin=dict(l=40, r=40, t=10, b=40),
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='closest'
        )
    }
    
    default_distance_plot = {
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
            hovermode='closest'
        )
    }
    
    return default_position_plot, default_distance_plot, "0.00", "0.00", "0", "0.0"

# Define anchor positions (same as in the server)
ANCHOR_POSITIONS = {
    'anchor1': (0, 0),
    'anchor2': (10, 0),
    'anchor3': (10, 10),
    'anchor4': (0, 10)
}

if __name__ == '__main__':
    print("Starting Wearable Positioning System Dashboard")
    print("Dashboard will connect to latest data from the server")
    app.run(debug=True, port=8050)
