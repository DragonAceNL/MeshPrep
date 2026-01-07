# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Command-line interface for MeshPrep."""

import click
from pathlib import Path
import logging

from meshprep.core import Mesh, ActionRegistry
from meshprep.core.repair_engine import RepairEngine
from meshprep.learning import HistoryTracker, StrategyLearner


@click.group()
@click.version_option(version="5.0.0")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose):
    """MeshPrep v5 - Automated mesh repair system."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output file (default: input_fixed.stl)")
@click.option("--pipeline", "-p", help="Pipeline to use (default: auto-select)")
@click.option("--no-ml", is_flag=True, help="Disable ML prediction")
@click.option("--no-learning", is_flag=True, help="Disable learning system")
def repair(input_file, output, pipeline, no_ml, no_learning):
    """Repair a mesh file."""
    input_path = Path(input_file)
    
    if output is None:
        output_path = input_path.parent / f"{input_path.stem}_fixed{input_path.suffix}"
    else:
        output_path = Path(output)
    
    click.echo(f"Repairing: {input_path}")
    
    # Setup tracker
    tracker = None if no_learning else HistoryTracker()
    
    # Setup predictor
    predictor = None
    if not no_ml:
        try:
            from meshprep.ml import PipelinePredictor
            predictor = PipelinePredictor()  # Would load trained model
            click.echo("✓ ML prediction enabled")
        except Exception:
            click.echo("⚠ ML not available, using learned statistics")
    
    # Create engine
    engine = RepairEngine(tracker=tracker)
    
    # Repair
    try:
        result = engine.repair(input_path)
        
        if result.success:
            result.mesh.trimesh.export(str(output_path))
            click.echo(f"✓ Repair successful: {output_path}")
            click.echo(f"  Pipeline: {result.pipeline_used}")
            click.echo(f"  Duration: {result.duration_ms:.1f}ms")
            if result.validation:
                click.echo(f"  Quality: {result.validation.quality_score}/5")
        else:
            click.echo(f"✗ Repair failed: {result.error}", err=True)
            raise click.Abort()
    
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option("--pipeline", "-p", help="Filter by pipeline name")
@click.option("--limit", "-n", default=10, help="Number of results to show")
def stats(pipeline, limit):
    """View repair statistics."""
    tracker = HistoryTracker()
    learner = StrategyLearner(tracker)
    
    if pipeline:
        # Show specific pipeline stats
        stats = tracker.get_pipeline_stats(pipeline)
        if stats:
            click.echo(f"\nPipeline: {pipeline}")
            click.echo(f"  Total attempts: {stats['total_attempts']}")
            click.echo(f"  Successes: {stats['successes']}")
            click.echo(f"  Success rate: {stats['successes']/stats['total_attempts']:.1%}")
            if stats['avg_quality']:
                click.echo(f"  Avg quality: {stats['avg_quality']:.2f}/5")
            if stats['avg_duration_ms']:
                click.echo(f"  Avg duration: {stats['avg_duration_ms']:.0f}ms")
        else:
            click.echo(f"No statistics for pipeline: {pipeline}")
    else:
        # Show summary
        summary = learner.get_statistics_summary()
        
        click.echo("\n📊 Repair Statistics")
        click.echo("=" * 50)
        click.echo(f"Total repairs: {summary['total_attempts']}")
        click.echo(f"Successes: {summary['total_successes']}")
        click.echo(f"Success rate: {summary['overall_success_rate']:.1%}")
        
        # Top pipelines
        click.echo(f"\n🏆 Top {limit} Pipelines:")
        recommendations = learner.recommend_pipelines(top_k=limit)
        for i, (name, score) in enumerate(recommendations, 1):
            click.echo(f"  {i}. {name}: {score:.3f}")
        
        # Suggestions
        suggestions = learner.suggest_improvements()
        if suggestions:
            click.echo("\n💡 Suggestions:")
            for suggestion in suggestions[:3]:
                click.echo(f"  • {suggestion}")


@cli.command()
def list_actions():
    """List available repair actions."""
    actions = ActionRegistry.list_actions()
    
    click.echo("\n📋 Available Actions")
    click.echo("=" * 50)
    
    for name, action in sorted(actions.items()):
        risk_icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(
            action.risk_level.value, "⚪"
        )
        click.echo(f"{risk_icon} {name:25} [{action.risk_level.value:6}] {action.description}")
    
    click.echo(f"\nTotal: {len(actions)} actions")


if __name__ == "__main__":
    cli()
