from collections import Counter
from operator import itemgetter
import datetime

import streamlit as st

import numpy as np

import plotly.express as px
import plotly.graph_objects as go

import vaex

from wordcloud import WordCloud

from actor_codes import actor_codes

# Turn cache on
if not vaex.cache.is_on():
    vaex.cache.on()

# Load the data
df = vaex.open('/data/gdelt/events_v2_streamlit.hdf5')
df = df._future()

# Build up the filter
def create_filter(codes, date_min, date_max):
    filter = (df.Actor1Type1Code.isin(codes) |
              df.Actor1Type2Code.isin(codes) |
              df.Actor1Type3Code.isin(codes) |
              df.Actor2Type1Code.isin(codes) |
              df.Actor2Type2Code.isin(codes) |
              df.Actor2Type3Code.isin(codes))
    if date_min is not None:
        filter = filter & (df.Date >= date_min)
    if date_max is not None:
        filter = filter & (df.Date <= date_max)

    return filter


# Compute all the relevant data
def compute_data(filter, binner_resolution, progress_function=None):
    # Filter the data
    dff = df.filter(filter)

    ## Aggregators for the global (worldwide trackers)
    aggs_global = {'mean_avg_tone': vaex.agg.mean(dff.AvgTone),
                   'std_avg_tone': vaex.agg.std(dff.AvgTone),
                   'mean_goldstein_scale': vaex.agg.mean(dff.GoldsteinScale),
                   'std_goldstein_scale': vaex.agg.std(dff.GoldsteinScale)}

    # Aggregators per country
    aggs_country = {'counts': 'count',
                    'avg_tone_sum': vaex.agg.sum(dff.AvgTone),
                    'goldstein_scale_sum': vaex.agg.sum(dff.GoldsteinScale),
                    'num_articles': vaex.agg.sum(dff.NumArticles),
                    'num_sources': vaex.agg.sum(dff.NumSources)}

    # Combine the country results
    aggs_country_combine = {'avg_tone': vaex.agg.sum('avg_tone_sum') / vaex.agg.sum('counts'),
                            'avg_tone': vaex.agg.sum('avg_tone_sum') / vaex.agg.sum('counts'),
                            'goldstein_scale': vaex.agg.sum('goldstein_scale_sum') / vaex.agg.sum('counts'),
                            'num_events': vaex.agg.sum('counts'),
                            'num_articles': vaex.agg.sum('num_articles'),
                            'num_sources': vaex.agg.sum('num_sources')}
    main_tree = vaex.progress.tree(progress_function)
    progress_groupby = main_tree.add("groupby")
    progress_agg = main_tree.add("agg")


    # Do the main operations, optimized pass over the data
    with progress_groupby:
        # The global single value summary stats
        total_events = dff.count(delay=True)
        avg_stats = dff.mean([dff.AvgTone, dff.GoldsteinScale], delay=True)
        total_stats = dff.sum([dff.NumSources, dff.NumArticles], delay=True)

        # Groupby per some time interval to plot the evolution of the tone and goldstein scale
        gdf = dff.groupby(vaex.BinnerTime(dff.Date, resolution=binner_resolution[0]), delay=True)


        # Groupby per country. There are two country codes (for each actor) so we do this twice and merge the results
        gdfc1 = dff.groupby(dff.Actor1CountryCode, delay=True)
        gdfc2 = dff.groupby(dff.Actor2CountryCode, delay=True)

        # Actor names - for the world cloud
        actor_names1 = dff.Actor1Name.value_counts(dropna=True, delay=True)
        actor_names2 = dff.Actor2Name.value_counts(dropna=True, delay=True)

        # Execute!
        dff.execute()

    # Gather the results of the computational graph
    # Global single value summary stats
    avg_tone, goldstein_scale = avg_stats.get()
    total_sources, total_articles = total_stats.get()

    with progress_agg:
        # Stats aggregated temporally
        gdf = gdf.get().agg(aggs_global)

        # Stats aggregated per country
        gdfc1 = gdfc1.get().agg(aggs_country)
        gdfc2 = gdfc2.get().agg(aggs_country)

    gdfc1.rename('Actor1CountryCode', 'CountryCode');
    gdfc2.rename('Actor2CountryCode', 'CountryCode');

    gdfc = vaex.concat((gdfc1, gdfc2))

    gdfc = gdfc.groupby('CountryCode').agg(aggs_country_combine)
    gdfc = gdfc.dropna(['CountryCode'])

    # Combine the two value counts result - a single dict of actor codes
    actor_names = Counter(actor_names1.get().to_dict()) + Counter(actor_names2.get().to_dict())
    del actor_names['missing']
    actor_names = dict(sorted(actor_names.items(), key = itemgetter(1), reverse = True)[:300])

    return avg_tone, goldstein_scale, total_events.get(), total_sources, total_articles, gdf, gdfc, actor_names


def create_line_plot(df, x, y, y_err, ylabel=None):
    '''
    :param df: a Vaex DataFrame
    :param x: an Expression to plot on the X axis
    :param y: an Expression to plot on the Y axis
    :param y_err: an Expression for the error (uncertainty) of the Y axis values
    :param ylabel: The label on the Y axis
    '''
    # Set the hovertemplate style
    hovertemplate = '<br> Date: %{x} <br> Value: %{y:.2f} ±%{customdata:.2f}<extra></extra>'

    # The range of the yaxis
    _mean_mm, _std_mm = df[y].minmax(), df[y_err].minmax()
    ylim = np.array([_mean_mm[0] - _std_mm[0], _mean_mm[1] + _std_mm[1]]) * 1.5

    # Get the data in a format Plotly accepts
    x = df[x].tolist()
    y = df[y].to_numpy()
    y_err = df[y_err].to_numpy()

    # The location of the error line (wrapping upon itself)
    y_err = (y + y_err).tolist() + (y - y_err).tolist()[::-1]


    # The traces...
    trace_mean = go.Scatter(x=x, y=y, customdata=y_err,
                            hovertemplate=hovertemplate,
                            showlegend=False)

    trace_std = go.Scatter(x=x + x[::-1], y=y_err,
                           fill='toself', fillcolor='rgba(0, 100, 80, 0.2)',
                           line=go.scatter.Line(width=0),
                           hoverinfo='skip',
                           showlegend=False)
    # The layout
    layout = go.Layout(xaxis=go.layout.XAxis(title='Date'),
                       yaxis=go.layout.YAxis(title=ylabel, range=ylim),
                       margin=go.layout.Margin(l=0, r=0, b=0, t=0),
                       height=300,
                       )

    return go.Figure(data=[trace_mean, trace_std], layout=layout)


def create_world_map(df):
    fig = px.choropleth(data_frame=df.to_pandas_df(),
              locations='CountryCode',
              color='avg_tone',
              color_continuous_scale='viridis_r',
              hover_data=['num_events', 'num_articles', 'num_sources', 'goldstein_scale'])

    hovertempate ='''<b>Country: %{location}</b><br>

    <br>Total events: %{customdata[0]:.3s}
    <br>Total articles: %{customdata[1]:.3s}
    <br>Total sources: %{customdata[2]:.3s}
    <br>Mean Tone: %{z:.2f}
    <br>Mean Goldstein scale: %{customdata[3]:.2f}
    '''
    with fig.batch_update():
        fig.update_layout(coloraxis_showscale=False)
        fig.update_layout(width=1000)
        fig.update_layout(margin=go.layout.Margin(l=0, r=0, b=0, t=0),)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_traces(hovertemplate=hovertempate)
        fig.update_layout(geo=go.layout.Geo(projection=go.layout.geo.Projection(type='natural earth')))
        fig.update_layout(coloraxis_showscale=False)
    return fig


def create_wordcloud(actor_names):
    wordcloudmaker = WordCloud(background_color='white',
                               width=1200,
                               height=900,
                               max_words=len(actor_names))
    wc_data = wordcloudmaker.generate_from_frequencies(actor_names)

    # Display the wordcloud
    fig = px.imshow(wc_data)
    with fig.batch_update():
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.layout['margin'] = {"r": 0, "t": 0, "l": 0, "b": 0}
        fig.data[0]['hoverinfo'] = 'skip'
        fig.data[0]['hovertemplate'] = None
    return fig


def human_format(num):
    '''Better formatting of large numbers
    Kudos to:
    '''
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


def get_actor_code_descriptions(codes):
    x = ''
    for code in codes:
        x += f' - {code}: {actor_codes.get(code)} \n'
    return x


def show_page():

    # Additional options for the sidebar
    # Choose actor codes
    codes = st.sidebar.multiselect(
        label='Select Actor Types',
        default='EDU',
        options=list(actor_codes.keys()),
        help='Select one ore more Actor Type codes.')

    # Specify date range
    date_range = st.sidebar.slider(
        label='Date Range',
        min_value=datetime.date(2014, 2, 18),
        max_value=datetime.date(2022, 4, 2),
        value=(datetime.date(2014, 2, 18), datetime.date(2022, 4, 2)),
        step=datetime.timedelta(days=1),
        help='Select a date range.')

    # Specify time resolution
    binner_resolution = st.sidebar.selectbox(label='Time Resolution', options=['Day', 'Week', 'Month', 'Year'], index=1)

    # Show a progress bar
    progress = st.sidebar.progress(0.0)

    def _progress_function(value):
        '''Wrapper to make the progress bar work with Vaex.'''
        progress.progress(value)
        return True

    # Reformat the date_range
    date_min = date_range[0].strftime('%Y-%m-%d')
    date_max = date_range[1].strftime('%Y-%m-%d')
    if date_min == '2014-02-18':
        date_min = None
    if date_max == '2022-04-02':
        date_max = None

    st.title('GDELT Actor Explorer')

    if len(codes) > 0:

        st.subheader('Actor types selected')
        st.markdown(get_actor_code_descriptions(codes))

        # Compute the filter
        filter = create_filter(codes, date_min, date_max)
        # Compute all relevant data needed for visualisation
        data = compute_data(filter=filter, binner_resolution=binner_resolution, progress_function=_progress_function)

        # The visualisation of the data starts here

        # Plot the global single value summary stats
        avg_tone, goldstein_scale, total_events, total_sources, total_articles, gdf, gdfc, actor_names = data

        st.subheader('Summary statistics')
        metric_cols = st.columns(5)
        metric_cols[0].metric(label='Events', value=human_format(total_events))
        metric_cols[1].metric(label='Articles', value=human_format(total_articles))
        metric_cols[2].metric(label='Sources', value=human_format(total_sources))
        metric_cols[3].metric(label='Avg. Tone', value=f'{avg_tone:.2f}')
        metric_cols[4].metric(label='Goldstein Scale', value=f'{goldstein_scale:.2f}')

        col_left, col_right = st.columns(2)
        col_left.subheader(f'Average Tone per {binner_resolution.lower()}')
        col_left.plotly_chart(create_line_plot(gdf, 'Date', 'mean_avg_tone', 'std_avg_tone'),
                              use_container_width=True)

        col_right.subheader(f'Goldstein scale per {binner_resolution.lower()}')
        col_right.plotly_chart(create_line_plot(gdf, 'Date', 'mean_goldstein_scale', 'std_goldstein_scale'),
                               use_container_width=True)

        st.subheader('Event statistics per Country')
        st.plotly_chart(create_world_map(gdfc), use_container_width=True)

        st.subheader('Actor names wordcloud')
        st.plotly_chart(create_wordcloud(actor_names), use_container_width=True)

    else:
        st.error('No actor codes selected. Please select at least one actor code.')
        st.stop()
