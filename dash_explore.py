import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

data = pd.read_csv('train.csv')


def generate_table(dataframe, max_rows=10):
    col_list = dataframe.columns
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in col_list])] +

        # Body
        [html.Tr([
            html.Td(dataframe.loc[i, col]) for col in col_list
        ]) for i in range(min(len(dataframe), max_rows))]
    )


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

target = data['Survived']
descriptive_data = data.loc[:, data.columns != 'Survived']

app.layout = html.Div(children=[
    html.H4(children='Titanic data'),
    generate_table(data),
    html.H4(children='Bivariate exploration'),
    html.Div([
        dcc.Dropdown(
            id='feature-select-dropdown',
            options=[{'label': i, 'value': i}
                     for i in descriptive_data.columns],
            value='Pclass'
        ),
        dcc.Graph(id='feature-target-graph')
    ])
])


@app.callback(
    dash.dependencies.Output('feature-target-graph', 'figure'),
    [dash.dependencies.Input('feature-select-dropdown', 'value')]
)
def update_graph(column_name):
    if column_name in ['Pclass', 'Sex', 'Embarked', 'Name']:
        return {
            'data': [go.Bar(
                x=descriptive_data[column_name].unique(),
                y=descriptive_data.loc[
                    target == i, column_name
                ].value_counts(),
                name=('Survived' if i else 'Not Survived')
            ) for i in target.unique()],
            'layout': go.Layout(
                barmode='group',
                title='Survival vs '+column_name
            )
        }
    else:
        return {
            'data': [go.Histogram(
                x=descriptive_data.loc[target == i, column_name],
                opacity=0.75,
                name=('Survived' if i else 'Not Survived')
            ) for i in target.unique()],
            'layout': go.Layout(
                barmode='overlay',
                title='Survival vs '+column_name
            )
        }


if __name__ == '__main__':
    app.run_server(debug=True)
