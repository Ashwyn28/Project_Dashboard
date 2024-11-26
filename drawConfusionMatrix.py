import plotly.figure_factory as ff
import pipeline
from sklearn.metrics import confusion_matrix

# y_test = pipeline.y_test
# y_pred = pipeline.y_pred

def cm(y_test, y_pred):

    # True negative, False positive, False negative, True positive values

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Matrix of values

    z = [
        [fn, tp],
        [tn, fp]
        ]

    x = ["Not Fatigued", "Fatigued"]
    y = ["Fatigued", "Not Fatigued"]

    z_text = [[str(y) for y in x] for x in z]

    # Creating figure

    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')

    fig.add_annotation(dict(font=dict(color="black", size=14),
                                x=0.5,
                                y=-0.15,
                                showarrow=False,
                                text="Predicted value"),
                                xref="paper",
                                yref="paper")

    fig.add_annotation(dict(font=dict(color="black", size=14), 
                            x = -0.35,
                            y=0.5,
                            showarrow=False,
                            text="Real value",
                            xref="paper",
                            yref="paper"))

    fig.update_layout(margin=dict(t=50, l=200))

    fig['data'][0]['showscale'] = True

    return fig




