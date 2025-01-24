import graphviz
import streamlit as st

def create_job_search_architecture():
    # Create a directed graph
    graph = graphviz.Digraph(format='png')

    # Define node styles
    styles = {
        'docker': {'shape': 'box', 'fillcolor': '#2496ED', 'fontcolor': 'white', 'penwidth': '3.0'},
        'start': {'shape': 'cylinder', 'style': 'filled', 'fillcolor': '#22c55e', 'fontcolor': 'white'},
        'input': {'shape': 'box', 'style': 'filled', 'fillcolor': '#f97316', 'fontcolor': 'white'},
        'decision': {'shape': 'diamond', 'style': 'filled', 'fillcolor': '#0ea5e9', 'fontcolor': 'white'},
        'process': {'shape': 'box', 'style': 'filled', 'fillcolor': '#8b5cf6', 'fontcolor': 'white'},
        'output': {'shape': 'box', 'style': 'filled', 'fillcolor': '#ec4899', 'fontcolor': 'white'}
    }

    # Create a subgraph for the Docker container to encapsulate everything
    with graph.subgraph(name='cluster_docker_deployment') as docker_cluster:
        docker_cluster.attr(
            label='🐳 Docker Deployment', 
            style='filled', 
            color='lightgrey', 
            **{k: v for k, v in styles['docker'].items() if k != 'style'}
        )

        # External User Interaction
        docker_cluster.node('Start', '🚀 Start\nJob Search App', **styles['start'])
        docker_cluster.node('JobInput', '💼 Input Job\nDescription', **styles['input'])
        docker_cluster.node('ResumeUpload', '📄 Upload\nResume', **styles['input'])
        docker_cluster.node('ApiKey', '🔑 GROQ\nAPI Key', **styles['input'])
        docker_cluster.node('ChooseAction', '🔄 Choose\nAction', **styles['decision'])

        # Output Nodes
        docker_cluster.node('JobListings', 'Job\nListings\n💼', **styles['output'])
        docker_cluster.node('MatchResults', 'Match\nResults\n✅', **styles['output'])

        # Internal Microservices
        docker_cluster.node('frontend', 'Streamlit\nFrontend\n🖥️', **styles['process'])
        docker_cluster.node('job_scraper', 'Multi-Site\nJob Scraper\n🌐', **styles['process'])
        docker_cluster.node('ai_service', 'AI Matching\nEngine\n🤖', **styles['process'])
        docker_cluster.node('database', 'Job & User\nDatabase\n💾', **styles['process'])

        # External Flow Connections
        docker_cluster.edge('Start', 'JobInput')
        docker_cluster.edge('Start', 'ResumeUpload')
        docker_cluster.edge('JobInput', 'ApiKey')
        docker_cluster.edge('ResumeUpload', 'ApiKey')
        docker_cluster.edge('ApiKey', 'ChooseAction')
        docker_cluster.edge('ChooseAction', 'frontend')
        docker_cluster.edge('frontend', 'JobListings')
        docker_cluster.edge('frontend', 'MatchResults')

        # Docker Internal Connections
        docker_cluster.edges([
            ('frontend', 'job_scraper'),
            ('frontend', 'ai_service'),
            ('ai_service', 'database'),
            ('job_scraper', 'database')
        ])

    # Graph settings
    graph.attr(rankdir='TB', splines='ortho')

    return graph