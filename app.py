"""
Okataina Volcanic Deformation Dashboard
Interactive modeling tool for volcano monitoring and unrest assessment
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.title = "Okataina Volcanic Deformation"


# ============================================================================
# McTigue 2D Finite Sphere Deformation Model
# ============================================================================
def mctigue2D(x0, y0, z0, P_G, a, nu, x, y):
    """
    Calculate surface displacement from a finite sphere source (McTigue 1987)
    
    Parameters:
    -----------
    x0, y0 : float
        Source location (Easting, Northing) in meters
    z0 : float
        Source depth in meters (positive down)
    P_G : float
        Normalized pressure change (ΔP/G, dimensionless)
    a : float
        Source radius in meters
    nu : float
        Poisson's ratio (typically 0.25)
    x, y : array_like
        Observation points (Easting, Northing) in meters
        
    Returns:
    --------
    u, v, w : array_like
        Displacement components (East, North, Up) in meters
    dwdx, dwdy : array_like
        Ground tilt components (radians)
    """
    # Translate coordinates to system centered at source
    xxn = x - x0
    yyn = y - y0
    
    # Radial distance from source center
    r = np.sqrt(xxn**2 + yyn**2)
    
    # Dimensionless coordinates
    csi = xxn / z0
    psi = yyn / z0
    rho = r / z0
    e = a / z0
    
    # Constants used in formulas
    f1 = 1.0 / (rho**2 + 1)**1.5
    f2 = 1.0 / (rho**2 + 1)**2.5
    c1 = e**3 / (7 - 5*nu)
    
    # Dimensionless displacement [McTigue (1987), eq. (52) and (53)]
    uzbar = (e**3) * (1-nu) * f1 * (1 - c1 * (0.5*(1+nu) - 3.75*(2-nu)/(rho**2+1)))
    urbar = rho * uzbar
    
    # Dimensional displacement
    # Handle r=0 case to avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        u = urbar * P_G * z0 * xxn / r  # East
        v = urbar * P_G * z0 * yyn / r  # North
        u = np.where(r == 0, 0, u)
        v = np.where(r == 0, 0, v)
    
    w = uzbar * P_G * z0  # Up
    
    # Ground tilt
    dwdx = -(1-nu) * P_G * csi * (e**3) * f2 * (3 - c1*(1.5*(1+nu) - 18.75*(2-nu)/(rho**2+1)))
    dwdy = -(1-nu) * P_G * psi * (e**3) * f2 * (3 - c1*(1.5*(1+nu) - 18.75*(2-nu)/(rho**2+1)))
    
    return u, v, w, dwdx, dwdy

# Load geographic data
def load_geographic_data():
    """Load lakes, calderas, and GPS stations"""
    #data_path = './data'
    data_path = os.path.join(os.path.dirname(__file__), 'data')
    
    # Load lakes
    lakes_df = pd.read_csv(os.path.join(data_path, 'OK_lakes_simple.csv'), 
                           header=None, names=['x', 'y', 'lake_id', 'point_id'])
    # Load calderas
    calderas_df = pd.read_csv(os.path.join(data_path, 'OK_calderas_simple.csv'),
                              header=None, names=['x', 'y', 'caldera_id', 'point_id'])
    
    # Load GPS stations
    stations_df = pd.read_csv(os.path.join(data_path, 'OK_stations.csv'), skiprows=1,
                             names=['x', 'y'])
    
    # Load station names
    with open(os.path.join(data_path, 'OK_station_names.txt'), 'r') as f:
        station_names = [line.strip() for line in f.readlines()]
    stations_df['name'] = station_names
    
    return lakes_df, calderas_df, stations_df

# Load data at startup
lakes_df, calderas_df, stations_df = load_geographic_data()

# Default parameters
DEFAULT_PARAMS = {
    'source_x': 1911757,  # Tarawera
    'source_y': 5764126,
    'depth': 6000,        # 6 km depth
    'radius': 2000,       # 2 km radius
    'pressure': 10e6,     # 10 MPa
}

# App layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🌋 Okataina Volcanic Deformation Dashboard", 
                style={'margin': '0', 'fontSize': '20px'}),
        html.P("Quick modeling tool for volcano monitoring and unrest assessment", 
               style={'fontSize': '12px', 'color': '#666', 'margin': '5px 0 0 0'})
    ], style={
        'textAlign': 'center', 
        'padding': '10px', 
        'backgroundColor': '#f8f9fa', 
        'borderBottom': '2px solid #dee2e6'
    }),
    
    html.Div([
        # Left panel - Controls
        html.Div([
            html.H3("Inflation Source", style={'color': '#2c3e50', 'marginBottom': '20px'}),
            
            # Source Type
            html.Div([
                html.Label("Source Type:", style={'fontWeight': 'bold'}),
                dcc.RadioItems(
                    id='source-type',
                    options=[
                        {'label': ' Magma Chamber (Sphere)', 'value': 'sphere'},
                        {'label': ' Dike (Vertical)', 'value': 'dike'},
                    ],
                    value='sphere',
                    style={'marginBottom': '20px'},
                    labelStyle={'display': 'block', 'marginBottom': '8px'}
                ),
            ], style={
                'marginBottom': '25px', 
                'padding': '15px', 
                'backgroundColor': '#fff', 
                'borderRadius': '5px', 
                'border': '1px solid #ddd'
            }),
            
            html.Hr(),
            
            # Location
            html.H4("Location", style={'marginTop': '20px', 'color': '#495057'}),
            html.Label("Quick Presets:"),
            dcc.Dropdown(
                id='location-preset',
                options=[
                    {'label': 'Tarawera', 'value': 'tarawera'},
                    {'label': 'Haroharo', 'value': 'haroharo'},
                    {'label': 'Custom', 'value': 'custom'},
                ],
                value='tarawera',
                style={'marginBottom': '15px'}
            ),
            
            html.Div(id='custom-location', children=[
                html.Label("Easting (m):"),
                dcc.Input(
                    id='source-x', 
                    type='number', 
                    value=DEFAULT_PARAMS['source_x'],
                    style={'width': '100%', 'marginBottom': '10px'}
                ),
                
                html.Label("Northing (m):"),
                dcc.Input(
                    id='source-y', 
                    type='number', 
                    value=DEFAULT_PARAMS['source_y'],
                    style={'width': '100%', 'marginBottom': '10px'}
                ),
            ]),
            
            html.Label("Depth (m):"),
            dcc.Slider(
                id='depth',
                min=1000, 
                max=15000, 
                step=500,
                value=DEFAULT_PARAMS['depth'],
                marks={i: f'{i/1000:.0f}km' for i in range(1000, 16000, 2000)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            
            html.Hr(style={'margin': '25px 0'}),
            
            # Source parameters - all sliders always exist, visibility controlled by callbacks
            html.Div([
                # Sphere parameters
                html.Div([
                    html.H4("Chamber Properties", style={'color': '#495057'}),
                    
                    html.Label("Radius (m):"),
                    dcc.Slider(
                        id='radius',
                        min=500, max=5000, step=100,
                        value=DEFAULT_PARAMS['radius'],
                        marks={i: f'{i/1000:.1f}km' for i in range(500, 5001, 1000)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Br(),
                    
                    html.Label("Pressure Change (MPa):"),
                    dcc.Slider(
                        id='pressure',
                        min=1, max=50, step=1,
                        value=10,
                        marks={i: f'{i}' for i in range(0, 51, 10)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], id='sphere-params', style={'display': 'block'}),
                
                # Dike parameters
                html.Div([
                    html.H4("Dike Properties", style={'color': '#495057'}),
                    
                    html.Label("Strike (degrees from North):"),
                    dcc.Slider(
                        id='strike',
                        min=0, max=180, step=5,
                        value=55,
                        marks={i: f'{i}°' for i in range(0, 181, 45)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Br(),
                    
                    html.Label("Opening (m):"),
                    dcc.Slider(
                        id='opening',
                        min=0.1, max=5, step=0.1,
                        value=1.0,
                        marks={i: f'{i}m' for i in range(0, 6)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], id='dike-params', style={'display': 'none'}),
            ]),
            
            # Interval component to trigger updates periodically
            dcc.Interval(
                id='update-interval',
                interval=2000,  # Update every 2000ms
                n_intervals=0
            ),
            
            html.Hr(style={'margin': '25px 0'}),
            
            # Info box
            html.Div(id='info-box', style={
                'marginTop': '20px',
                'padding': '15px',
                'backgroundColor': '#e7f3ff',
                'borderRadius': '5px',
                'border': '1px solid #b3d9ff',
                'fontSize': '13px'
            }),
            
        ], style={
            'width': '350px',
            'padding': '20px',
            'backgroundColor': '#f8f9fa',
            'borderRight': '2px solid #dee2e6',
            'height': '100vh',
            'overflowY': 'auto'
        }),
        
        # Right panel - Map
        html.Div([
            dcc.Tabs(id='tabs', value='map', children=[
                dcc.Tab(label='📍 Map View', value='map', children=[
                    dcc.Graph(id='map-plot', style={'height': '650px'})
                ]),
                dcc.Tab(label='📊 Displacement', value='displacement', children=[
                    dcc.Graph(id='displacement-plot', style={'height': '650px'})
                ]),
                dcc.Tab(label='📈 Profile', value='profile', children=[
                    dcc.Graph(id='profile-plot', style={'height': '650px'})
                ]),
            ], style={'marginBottom': '0'}),
        ], style={'flex': '1', 'padding': '0'}),
        
    ], style={'display': 'flex', 'flexDirection': 'row'}),
])


# Callbacks
@app.callback(
    Output('custom-location', 'style'),
    Input('location-preset', 'value')
)
def toggle_custom_location(preset):
    """Show/hide custom location inputs"""
    if preset == 'custom':
        return {'display': 'block'}
    return {'display': 'none'}


@app.callback(
    [Output('source-x', 'value'),
     Output('source-y', 'value')],
    Input('location-preset', 'value'),
    prevent_initial_call=True
)
def update_location_preset(preset):
    """Update location based on preset"""
    presets = {
        'tarawera': (1911757, 5764126),
        'haroharo': (1907441, 5778217),
    }
    if preset in presets:
        return presets[preset]
    return dash.no_update, dash.no_update


@app.callback(
    [Output('sphere-params', 'style'),
     Output('dike-params', 'style')],
    Input('source-type', 'value')
)
def update_source_params_visibility(source_type):
    """Show/hide parameter sections based on source type"""
    if source_type == 'sphere':
        return {'display': 'block'}, {'display': 'none'}
    else:  # dike
        return {'display': 'none'}, {'display': 'block'}


def create_base_map(x0=None, y0=None, displacements=None):
    """
    Create base map with lakes, calderas, and GPS stations in NZTM coordinates
    
    Parameters:
    -----------
    x0, y0 : float, optional
        Source location
    displacements : dict, optional
        Dictionary with 'u', 'v', 'w' keys containing displacement arrays
    """
    fig = go.Figure()
    
    # Group lakes by lake_id and plot as polygons
    lake_ids = sorted(lakes_df['lake_id'].unique())
    for i, lake_id in enumerate(lake_ids):
        lake_points = lakes_df[lakes_df['lake_id'] == lake_id].sort_values('point_id')
        fig.add_trace(go.Scatter(
            x=lake_points['x'],
            y=lake_points['y'],
            mode='lines',
            fill='toself',
            fillcolor='rgba(100, 180, 255, 0.3)',
            line=dict(color='blue', width=1),
            name='Lakes',
            showlegend=bool(i == 0),
            legendgroup='lakes',
            hoverinfo='skip'
        ))
    
    # Group calderas by caldera_id and plot as polygons
    caldera_ids = sorted(calderas_df['caldera_id'].unique())
    for i, caldera_id in enumerate(caldera_ids):
        caldera_points = calderas_df[calderas_df['caldera_id'] == caldera_id].sort_values('point_id')
        fig.add_trace(go.Scatter(
            x=caldera_points['x'],
            y=caldera_points['y'],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Calderas',
            showlegend=bool(i == 0),
            legendgroup='calderas',
            hoverinfo='skip'
        ))
    
    # Plot GPS stations
    fig.add_trace(go.Scatter(
        x=stations_df['x'],
        y=stations_df['y'],
        mode='markers+text',
        marker=dict(
            size=10,
            color='black',
            symbol='triangle-up',
            line=dict(color='white', width=1)
        ),
        text=stations_df['name'],
        textposition='top center',
        textfont=dict(size=10, color='black'),
        name='GPS Stations',
        hovertemplate='<b>%{text}</b><br>E: %{x:.0f}<br>N: %{y:.0f}<extra></extra>'
    ))
    
    # Plot displacement vectors (glyphs) if provided
    if displacements is not None:
        u = displacements['u']
        v = displacements['v']
        w = displacements['w']
        
        # Calculate displacement magnitudes
        mags = np.sqrt(u**2 + v**2)
        max_mag = np.max(mags)
        
        # Scale factor for visualization - needs to be LARGE to see at map scale
        # 1 mm displacement = 1 km on map
        scale = 1000000
        
        # Add arrows from station to station+displacement
        for i, (name, x, y) in enumerate(zip(stations_df['name'], stations_df['x'], stations_df['y'])):
            mag = mags[i]
            
            if mag > 1e-10:
                # Proportional arrowhead sizing with limits
                size_factor = (mag / max_mag) if max_mag > 0 else 0.5
                
                # Arrowhead size: scale from 0.8 (small) to max 3.0 (large)
                # This is in Plotly's relative units, not meters
                # Limit to 3.0 to prevent excessively large arrowheads
                arrowhead_size = 0.8 + 2.2 * size_factor  # 0.8 to 3.0
                arrowhead_size = min(arrowhead_size, 1.0)  # Hard cap at 3.0
                
                # Arrow width also proportional
                arrow_width = 1.5 + 2.5 * size_factor  # 1.5 to 4.0
                arrow_width = min(arrow_width, 4.0)  # Hard cap at 4.0
                
                fig.add_annotation(
                    x=x + u[i]*scale,
                    y=y + v[i]*scale,
                    ax=x,
                    ay=y,
                    xref='x',
                    yref='y',
                    axref='x',
                    ayref='y',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=arrowhead_size,
                    arrowwidth=arrow_width,
                    arrowcolor='darkgreen',
                    opacity=0.8,
                )
                
                # Add text showing displacement magnitude
                disp_mm = mag * 1000
                fig.add_trace(go.Scatter(
                    x=[x + u[i]*scale],
                    y=[y + v[i]*scale],
                    mode='text',
                    text=[f'{disp_mm:.1f}mm'],
                    textposition='top center',
                    textfont=dict(size=9, color='darkgreen'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Add legend entry
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=0),
            showlegend=True,
            name='Displacements (scaled)',
            legendgroup='displacements',
            hoverinfo='skip'
        ))
    
    # Add source location if provided
    if x0 is not None and y0 is not None:
        fig.add_trace(go.Scatter(
            x=[x0],
            y=[y0],
            mode='markers',
            marker=dict(
                size=15,
                color='red',
                symbol='star',
                line=dict(color='darkred', width=2)
            ),
            name='Source',
            hovertemplate='<b>Source</b><br>E: %{x:.0f}<br>N: %{y:.0f}<extra></extra>'
        ))
    
    # Update layout for NZTM cartesian map
    # Hardwired plot limits for Okataina region
    xok1, xok2 = 1868000, 1940000
    yok1, yok2 = 5730000, 5802000
    
    fig.update_layout(
        title=dict(
            text='Okataina Volcanic Centre (NZTM/EPSG:2193)',
            font=dict(size=18, color='#2c3e50')
        ),
        xaxis=dict(
            title='Easting (m)',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(150,150,150,0.3)',
            zeroline=False,
            tickformat=',d',  # Comma-separated thousands
            dtick=5000,  # Grid every 5km
            range=[xok1, xok2],  # Fixed boundaries
        ),
        yaxis=dict(
            title='Northing (m)',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(150,150,150,0.3)',
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,  # Equal aspect ratio for true cartesian
            tickformat=',d',  # Comma-separated thousands
            dtick=5000,  # Grid every 5km
            range=[yok1, yok2],  # Fixed boundaries
        ),
        hovermode='closest',
        plot_bgcolor='rgb(248, 248, 245)',  # Slightly warm background like a map
        paper_bgcolor='white',
        height=600,  # Further reduced to ensure it fits
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='rgb(180, 180, 180)',
            borderwidth=1
        ),
        margin=dict(l=60, r=20, t=60, b=50),  # Tighter margins, especially top
        # Add shapes for a subtle map border
        shapes=[
            dict(
                type='rect',
                xref='paper',
                yref='paper',
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(color='rgb(100, 100, 100)', width=2)
            )
        ]
    )
    
    return fig


@app.callback(
    [Output('map-plot', 'figure'),
     Output('displacement-plot', 'figure'),
     Output('profile-plot', 'figure'),
     Output('info-box', 'children')],
    [Input('source-type', 'value'),
     Input('source-x', 'value'),
     Input('source-y', 'value'),
     Input('depth', 'value'),
     Input('update-interval', 'n_intervals')],
    [State('radius', 'value'),
     State('pressure', 'value')],
    prevent_initial_call=False
)
def calculate_and_plot(source_type, x0, y0, z0, n_intervals, radius, pressure):
    """Main calculation and plotting callback - updates automatically when parameters change"""
    
    # Use default values if sliders don't exist yet (during initialization)
    if radius is None:
        radius = DEFAULT_PARAMS['radius']
    if pressure is None:
        pressure = 10  # MPa
    
    # Default parameters
    if source_type == 'sphere':
        # Use parameters from sliders
        shear_modulus = 3e10  # 30 GPa
        nu = 0.25
        
        # Convert pressure from MPa to Pa
        pressure_pa = pressure * 1e6
        
        # Calculate normalized pressure
        P_G = pressure_pa / shear_modulus
        
        # Calculate displacements at GPS stations
        station_x = stations_df['x'].values
        station_y = stations_df['y'].values
        
        u, v, w, dwdx, dwdy = mctigue2D(x0, y0, z0, P_G, radius, nu, station_x, station_y)
        
        # Store displacements
        displacements = {'u': u, 'v': v, 'w': w}
        
        # Create map with displacement glyphs
        map_fig = create_base_map(x0, y0, displacements)
        
        # Calculate max values for info
        max_uplift_mm = np.max(w) * 1000
        max_horizontal_mm = np.max(np.sqrt(u**2 + v**2)) * 1000
        
        # Info box
        info = html.Div([
            html.Strong("Results:"),
            html.Br(),
            f"• Max uplift: {max_uplift_mm:.2f} mm",
            html.Br(),
            f"• Max horizontal: {max_horizontal_mm:.2f} mm",
            html.Br(),
            f"• Source depth: {z0/1000:.1f} km",
            html.Br(),
            f"• Radius: {radius/1000:.1f} km",
            html.Br(),
            f"• Pressure: {pressure:.0f} MPa",
            html.Br(),
            html.Br(),
            html.Em("Map updates automatically when you change parameters")
        ])
        
    else:
        # Dike - not yet implemented
        displacements = None
        map_fig = create_base_map(x0, y0)
        
        info = html.Div([
            html.Strong("Dike model:"),
            html.Br(),
            html.Em("Coming soon!")
        ])
    
    # Placeholder for displacement plot (could show bar chart of station displacements)
    disp_fig = go.Figure()
    if displacements is not None:
        # Create bar chart of vertical displacements
        disp_fig.add_trace(go.Bar(
            x=stations_df['name'],
            y=w * 1000,  # Convert to mm
            marker_color='blue',
            name='Vertical'
        ))
        disp_fig.update_layout(
            title='Vertical Displacement at GPS Stations',
            xaxis_title='Station',
            yaxis_title='Uplift (mm)',
            showlegend=False,
            height=400
        )
    else:
        disp_fig.update_layout(
            title='Displacement data will appear after calculation',
            xaxis_title='Station',
            yaxis_title='Displacement (mm)'
        )
    
    # Profile plot - show uplift along E-W line through source
    profile_fig = go.Figure()
    if displacements is not None:
        # Create profile line
        profile_x = np.linspace(x0 - 30000, x0 + 30000, 100)
        profile_y = np.full_like(profile_x, y0)
        
        u_prof, v_prof, w_prof, _, _ = mctigue2D(x0, y0, z0, P_G, radius, nu, profile_x, profile_y)
        
        profile_fig.add_trace(go.Scatter(
            x=(profile_x - x0) / 1000,
            y=w_prof * 1000,
            mode='lines',
            line=dict(color='blue', width=2),
            name='Uplift'
        ))
        
        # Add station positions on profile
        for i, (name, sx, sy) in enumerate(zip(stations_df['name'], station_x, station_y)):
            if abs(sy - y0) < 5000:  # Within 5km of profile line
                profile_fig.add_trace(go.Scatter(
                    x=[(sx - x0) / 1000],
                    y=[w[i] * 1000],
                    mode='markers+text',
                    marker=dict(size=10, color='red'),
                    text=[name],
                    textposition='top center',
                    name=name,
                    showlegend=False
                ))
        
        profile_fig.update_layout(
            title='East-West Uplift Profile (through source)',
            xaxis_title='Distance from source (km)',
            yaxis_title='Vertical displacement (mm)',
            hovermode='x',
            height=400
        )
    else:
        profile_fig.update_layout(
            title='Profile will appear after calculation',
            xaxis_title='Distance from source (km)',
            yaxis_title='Vertical displacement (mm)'
        )
    
    return map_fig, disp_fig, profile_fig, info


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8050))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    app.run(debug=debug, host='0.0.0.0', port=port)
