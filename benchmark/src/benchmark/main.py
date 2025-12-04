#!/usr/bin/env python3

import asyncio
import os
import sys
from typing import Optional
import click
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set AWS credentials from environment
os.getenv('AWS_BEARER_TOKEN_BEDROCK')

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarker import Benchmarker
from utils import (
    create_output_directory,
    create_prompt_output_directory,
    save_results_csv,
    save_answers_json,
    save_prompt_info_json,
    validate_aws_credentials,
    print_summary_table,
    get_default_prompt,
    load_prompts_config,
    aggregate_results_by_model,
    save_aggregated_results_csv
)


@click.command()
@click.option(
    "--specific-model",
    type=str,
    help="Test only one specific model (use model name from config)"
)
@click.option(
    "--number-of-calls",
    type=int,
    default=100,
    help="Number of asynchronous calls per model (default: 100)"
)
@click.option(
    "--prompt",
    type=str,
    help="Custom prompt for testing (optional, uses configured prompts if not provided)"
)
@click.option(
    "--specific-prompt",
    type=str,
    help="Test only one specific prompt (use prompt name from config)"
)
@click.option(
    "--region",
    type=str,
    default="eu-central-1",
    help="AWS region for Bedrock (default: eu-central-1)"
)
def main(
    specific_model: Optional[str],
    number_of_calls: int,
    prompt: Optional[str],
    specific_prompt: Optional[str],
    region: str
):
    """
    AWS Bedrock LLM Benchmarking Framework
    
    A comprehensive benchmarking tool for testing AWS Bedrock LLMs with concurrent 
    load testing, performance metrics, error tracking, and cost analysis.
    """
    
    print("AWS Bedrock LLM Benchmarking Framework")
    print("=" * 50)
    
    # Validate AWS credentials
    if not validate_aws_credentials():
        print("❌ Error: AWS credentials not found or invalid.")
        print("   Please configure your AWS credentials using:")
        print("   - AWS CLI: aws configure")
        print("   - Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
        print("   - IAM roles (for EC2/Lambda)")
        sys.exit(1)
    
    print("✅ AWS credentials validated")
    
    # Determine if using custom prompt or configured prompts
    use_custom_prompt = prompt is not None
    if use_custom_prompt:
        print("📝 Using custom prompt")
    else:
        print("📝 Using configured prompts from prompts.json")
    
    # Validate number of calls
    if number_of_calls <= 0:
        print("❌ Error: Number of calls must be greater than 0")
        sys.exit(1)
    
    if number_of_calls > 1000:
        print("⚠️  Warning: Large number of calls may result in high costs and rate limiting")
        if not click.confirm("Do you want to continue?"):
            sys.exit(0)
    
    print(f"🚀 Configuration:")
    print(f"   - Region: {region}")
    print(f"   - Specific model: {specific_model or 'All models'}")
    print(f"   - Specific prompt: {specific_prompt or 'All prompts'}")
    print(f"   - Number of calls per model per prompt: {number_of_calls}")
    print()
    
    # Run the benchmark
    asyncio.run(
        run_benchmark_async(
            custom_prompt=prompt,
            number_of_calls=number_of_calls,
            specific_model=specific_model,
            specific_prompt=specific_prompt,
            region=region
        )
    )


async def run_benchmark_async(
    custom_prompt: Optional[str],
    number_of_calls: int,
    specific_model: Optional[str],
    specific_prompt: Optional[str],
    region: str
):
    try:
        # Initialize benchmarker
        benchmarker = Benchmarker(region_name=region)
        
        # Create output directory
        output_dir = create_output_directory()
        print(f"📁 Output directory: {output_dir}")
        print()
        
        # Run benchmark
        print("🔄 Starting benchmark...")
        
        if custom_prompt:
            # Single custom prompt mode
            prompt_info = {
                "name": "custom",
                "description": "Custom prompt provided via CLI",
                "prompt": custom_prompt
            }
            results = await benchmarker.run_single_prompt_benchmark(
                prompt=custom_prompt,
                prompt_info=prompt_info,
                number_of_calls=number_of_calls,
                specific_model=specific_model
            )
            all_results = results
            prompt_results = {"custom": results}
        else:
            # Multi-prompt mode using configured prompts
            benchmark_data = await benchmarker.run_multi_prompt_benchmark(
                number_of_calls=number_of_calls,
                specific_model=specific_model,
                specific_prompt=specific_prompt
            )
            all_results = benchmark_data["all_results"]
            prompt_results = benchmark_data["prompt_results"]
        
        if not all_results:
            print("❌ No benchmark results to save")
            return
        
        # Save results
        print("💾 Saving results...")
        
        # Aggregate results by model across all prompts for overall CSV
        aggregated_results = aggregate_results_by_model(all_results)
        overall_csv_path = os.path.join(output_dir, "results_overall.csv")
        save_aggregated_results_csv(aggregated_results, overall_csv_path)
        print(f"   - Saved aggregated overall results: {overall_csv_path}")
        
        # Save prompt-specific results
        for prompt_name, results in prompt_results.items():
            # Create prompt directory
            prompt_dir = create_prompt_output_directory(output_dir, prompt_name)
            
            # Save prompt info
            prompt_info = results[0]["prompt_info"] if results else {}
            prompt_info_path = os.path.join(prompt_dir, "prompt_info.json")
            save_prompt_info_json(prompt_info, prompt_info_path)
            
            # Save prompt-specific results CSV
            prompt_csv_path = os.path.join(prompt_dir, "prompt_results.csv")
            save_results_csv(results, prompt_csv_path)
            print(f"   - Saved {prompt_name} results: {prompt_csv_path}")
            
            # Save individual model responses as JSON
            for result in results:
                model_name = result["model_name"].replace(" ", "-").replace("/", "-")
                answers_json_path = os.path.join(prompt_dir, f"answers_{model_name}.json")
                save_answers_json(result, answers_json_path)
                print(f"   - Saved {prompt_name}/{model_name} responses: {answers_json_path}")
        
        # Print summary using aggregated results - convert format for print function
        formatted_results = []
        for result in aggregated_results:
            formatted_results.append({
                "model_name": result["model_name"],
                "metrics": result
            })
        print_summary_table(formatted_results)
        
        # Calculate and print total costs using aggregated results
        total_cost = sum(result["total_cost"] for result in aggregated_results)
        total_calls = sum(result["total_calls"] for result in aggregated_results)
        total_successful = sum(result["successful_calls"] for result in aggregated_results)
        
        print(f"\n📊 OVERALL SUMMARY:")
        print(f"   - Total prompts tested: {len(prompt_results)}")
        print(f"   - Total calls made: {total_calls:,}")
        print(f"   - Total successful calls: {total_successful:,}")
        print(f"   - Overall success rate: {(total_successful/total_calls*100):.1f}%")
        print(f"   - Total cost: ${total_cost:.6f}")
        print(f"   - Average cost per call: ${(total_cost/total_successful):.6f}" if total_successful > 0 else "   - Average cost per call: N/A")
        print(f"\n✅ Benchmark completed successfully!")
        print(f"   Results saved in: {output_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Benchmark failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()