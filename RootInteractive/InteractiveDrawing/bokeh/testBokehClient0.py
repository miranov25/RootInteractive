from bokeh.io import output_notebook, show, output_file
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import CustomJS, ColumnDataSource, Slider


def testBokehClient0():
    output_file("callback.html")
    x = [x * 0.05 for x in range(0, 201)]
    y = [a ** 3 - a ** 2 - 10 * a for a in x]

    s1 = ColumnDataSource(data=dict(x=x, y=y))
    s2 = ColumnDataSource(data=dict(x=x, y=y))

    #p = figure(plot_width=400, plot_height=400)
    #p.line('x', 'y', source=s1, line_width=3, line_alpha=0.6)

    p2 = figure(plot_width=400, plot_height=400)
    p2.line('x', 'y', source=s2, line_width=3, line_alpha=0.6)

    sliderMax = Slider(start=0.1, end=10, value=5, step=.1, title="Max")
    sliderMin = Slider(start=0.1, end=10, value=5, step=.1, title="Min")

    callback = CustomJS(args=dict(s1=s1, s2=s2, sliderMax=sliderMax, sliderMin=sliderMin), code="""
    var d1 = s1.data;
    var d2 = s2.data;
    var min = sliderMin.value;
    var max = sliderMax.value;
    d2['x'] = [];
    d2['y'] = [];
    var x = d1['x'];
    var y = d1['y'];

    for (i = 0; i < d1['x'].length; i++) {
        if (x[i]<max &&  x[i]>min){
            d2['x'].push(d1['x'][i]);
            d2['y'].push(d1['y'][i]);
        }
    }
    s2.change.emit();
    """)
    sliderMax.js_on_change('value', callback)
    sliderMin.js_on_change('value', callback)
    show(column(sliderMax, sliderMin, p2))



#testBokehClient0()
# customjsForSelections()
#test2()
