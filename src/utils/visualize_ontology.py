import json


INTERNAL_NODE_COLOR = "#2196f3"
LEAF_NODE_COLOR = "#4caf50"


def _truncate_desc(desc: str, limit: int = 100) -> str:
    desc = str(desc or "").strip()
    if len(desc) <= limit:
        return desc
    return desc[:limit].rstrip() + "..."


def _build_tooltip(node: dict) -> str:
    name = str(node.get("name", "Unnamed"))
    level = int(node.get("level", 0))
    docs = len(node.get("doc_ids", []) or [])
    desc = _truncate_desc(node.get("desc", ""))
    return (
        f"<b>{name}</b><br/>"
        f"Level: {level}<br/>"
        f"Docs: {docs}<br/>"
        f"Desc: {desc}"
    )


def _to_echarts_tree(node: dict) -> dict:
    children = node.get("children", []) or []
    doc_count = len(node.get("doc_ids", []) or [])
    is_leaf = not children
    name = str(node.get("name", "Unnamed"))

    if is_leaf:
        display_name = f"{name} ({doc_count} docs)"
        color = LEAF_NODE_COLOR
    else:
        display_name = name
        color = INTERNAL_NODE_COLOR

    return {
        "name": display_name,
        "value": doc_count,
        "collapsed": False,
        "children": [_to_echarts_tree(child) for child in children],
        "itemStyle": {"color": color},
        "tooltip": {"formatter": _build_tooltip(node)},
    }


def generate_html(json_path: str, html_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        root = json.load(f)

    data_json = json.dumps(_to_echarts_tree(root))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ontology Tree Visualization</title>
    <!-- Include ECharts -->
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
    <style>
        body, html {{
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            font-family: Arial, sans-serif;
            background-color: #f4f7f6;
        }}
        #main {{
            width: 100%;
            height: 100%;
        }}
        .header {{
            position: absolute;
            top: 20px;
            left: 20px;
            z-index: 10;
            background: rgba(255, 255, 255, 0.9);
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 20px;
            color: #333;
        }}
        .header p {{
            margin: 0;
            font-size: 14px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Ontology Tree Visualization</h1>
        <p>Interactive view of the ontology structure.</p>
        <p>Scroll to zoom, drag to pan. Click nodes to expand/collapse.</p>
    </div>
    <div id="main"></div>
    <script>
        var data = {data_json};

        var chartDom = document.getElementById('main');
        var myChart = echarts.init(chartDom);
        var option;

        myChart.showLoading();
        
        myChart.hideLoading();

        myChart.setOption(
            (option = {{
                tooltip: {{
                    trigger: 'item',
                    triggerOn: 'mousemove',
                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                    borderColor: '#ccc',
                    borderWidth: 1,
                    textStyle: {{
                        color: '#333'
                    }}
                }},
                series: [
                    {{
                        type: 'tree',
                        data: [data],
                        top: '5%',
                        left: '5%',
                        bottom: '5%',
                        right: '45%',
                        symbolSize: 8,
                        edgeShape: 'polyline',
                        edgeForkPosition: '50%',
                        initialTreeDepth: -1,
                        lineStyle: {{
                            width: 1,
                            color: '#e0e0e0'
                        }},
                        label: {{
                            backgroundColor: 'transparent',
                            position: 'right',
                            verticalAlign: 'middle',
                            align: 'left',
                            fontSize: 14,
                            padding: [2, 2],
                            color: '#333',
                            fontWeight: 'bold'
                        }},
                        leaves: {{
                            label: {{
                                position: 'right',
                                verticalAlign: 'middle',
                                align: 'left',
                                color: '#1a73e8',
                                fontWeight: 'normal'
                            }}
                        }},
                        expandAndCollapse: true,
                        roam: true,
                        layout: 'orthogonal',
                        animationDuration: 550,
                        animationDurationUpdate: 750
                    }}
                ]
            }})
        );

        window.addEventListener('resize', myChart.resize);
    </script>
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
