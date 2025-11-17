"""
Visualize client clustering assignments over rounds using a Sankey diagram.
"""
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

def visualize_clustering(experiment_name: str = "abl1"):
    """
    Generates a Sankey diagram visualization of client clustering assignments over rounds.

    Args:
        experiment_name (str): The name of the experiment (e.g., "abl1").
    
    사용 방법) python scripts/visualize_clustering.py --experiment abl1
    """
    log_file = Path(f"results/{experiment_name}/cluster_assignments.csv")
    if not log_file.exists():
        print(f"Error: Cluster assignment log file not found at {log_file}")
        print("Please run the experiment first to generate the log file.")
        return

    df = pd.read_csv(log_file)
    df.sort_values(by=['round', 'client_id'], inplace=True)

    all_nodes = []
    all_links = []
    
    # Create nodes
    for round_num in df['round'].unique():
        clusters = df[df['round'] == round_num]['cluster_id'].unique()
        for cluster_id in clusters:
            all_nodes.append({'round': round_num, 'cluster_id': cluster_id, 'id': f'R{round_num}-C{cluster_id}'})

    node_map = {node['id']: i for i, node in enumerate(all_nodes)}
    
    # Create links
    for round_num in range(df['round'].min(), df['round'].max()):
        df_round1 = df[df['round'] == round_num]
        df_round2 = df[df['round'] == round_num + 1]
        
        merged = pd.merge(df_round1, df_round2, on='client_id', suffixes=('_1', '_2'))
        
        links = merged.groupby(['cluster_id_1', 'cluster_id_2']).size().reset_index(name='value')
        
        for _, row in links.iterrows():
            source_id = f'R{round_num}-C{row["cluster_id_1"]}'
            target_id = f'R{round_num+1}-C{row["cluster_id_2"]}'
            
            if source_id in node_map and target_id in node_map:
                all_links.append({
                    'source': node_map[source_id],
                    'target': node_map[target_id],
                    'value': row['value']
                })

    if not all_nodes or not all_links:
        print("Not enough data to generate a Sankey diagram.")
        return

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=[node['id'] for node in all_nodes],
        ),
        link=dict(
            source=[link['source'] for link in all_links],
            target=[link['target'] for link in all_links],
            value=[link['value'] for link in all_links]
        ))])

    fig.update_layout(
        title_text=f"Client Clustering Flow Over Rounds ({experiment_name.upper()})",
        font_size=10
    )

    output_dir = Path("results/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{experiment_name}_cluster_sankey.html"
    
    fig.write_html(output_path)
    print(f"Sankey diagram saved to: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize client clustering assignments using a Sankey diagram.")
    parser.add_argument(
        "--experiment",
        type=str,
        default="abl1",
        help="The name of the experiment to visualize (e.g., 'abl1')."
    )
    args = parser.parse_args()
    
    visualize_clustering(args.experiment)