#!/bin/bash

# =============================================================================
# Progress Collector System - Complete Demo Runner
# Executes all framework demos and generates comprehensive analysis
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO:${NC} $1"
}

# Function to check if we're in the correct directory
check_directory() {
    if [[ ! -f "requirements.txt" || ! -d "src" || ! -d "samples" ]]; then
        error "Please run this script from the ai-ram-crewai-langchain project root directory"
        exit 1
    fi
}

# Function to check virtual environment
check_venv() {
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        warning "Virtual environment not detected. Activating venv..."
        if [[ -d "venv" ]]; then
            source venv/bin/activate
            log "Virtual environment activated"
        else
            error "Virtual environment not found. Please create one with: python -m venv venv"
            exit 1
        fi
    else
        log "Virtual environment detected: $VIRTUAL_ENV"
    fi
}

# Function to check dependencies
check_dependencies() {
    info "Checking required dependencies..."
    
    # Check if key packages are installed
    python -c "import pandas, openpyxl" 2>/dev/null || {
        error "Required packages not found. Installing from requirements.txt..."
        pip install -r requirements.txt
    }
    
    log "Dependencies check completed"
}

# Function to create output directory
setup_output() {
    local output_dir="samples/output"
    if [[ ! -d "$output_dir" ]]; then
        mkdir -p "$output_dir"
        log "Created output directory: $output_dir"
    fi
    
    # Archive old comprehensive reports
    if ls samples/output/comprehensive_* >/dev/null 2>&1; then
        local archive_dir="samples/output/archive_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$archive_dir"
        mv samples/output/comprehensive_* "$archive_dir/" 2>/dev/null || true
        info "Archived previous comprehensive reports to: $archive_dir"
    fi
}

# Function to display system info
show_system_info() {
    echo -e "${PURPLE}"
    echo "═══════════════════════════════════════════════════════════════"
    echo "🚀 PROGRESS COLLECTOR SYSTEM - COMPLETE DEMO RUNNER"
    echo "═══════════════════════════════════════════════════════════════"
    echo -e "${NC}"
    
    info "System Information:"
    echo "  • Date: $(date)"
    echo "  • Python: $(python --version 2>&1)"
    echo "  • Working Directory: $(pwd)"
    echo "  • Virtual Environment: ${VIRTUAL_ENV:-"Not activated"}"
    
    info "Available Demo Scripts:"
    echo "  • simple_test.py - Basic functionality test"
    echo "  • mock_demo.py - Mock processor demo (3 sample docs)"
    echo "  • comprehensive_demo.py - Full demo (all 16 documents)"
    echo "  • analyze_results.py - Results analysis"
    
    echo ""
}

# Function to run simple test
run_simple_test() {
    echo -e "${CYAN}"
    echo "═══════════════════════════════════════════════════════════════"
    echo "🧪 RUNNING SIMPLE TEST"
    echo "═══════════════════════════════════════════════════════════════"
    echo -e "${NC}"
    
    log "Starting simple functionality test..."
    
    if python simple_test.py; then
        log "✅ Simple test completed successfully"
    else
        error "❌ Simple test failed"
        return 1
    fi
    
    echo ""
}

# Function to run mock demo
run_mock_demo() {
    echo -e "${CYAN}"
    echo "═══════════════════════════════════════════════════════════════"
    echo "🎭 RUNNING MOCK DEMO"
    echo "═══════════════════════════════════════════════════════════════"
    echo -e "${NC}"
    
    log "Starting mock demo with sample documents..."
    
    if python mock_demo.py; then
        log "✅ Mock demo completed successfully"
    else
        error "❌ Mock demo failed"
        return 1
    fi
    
    echo ""
}

# Function to run comprehensive demo
run_comprehensive_demo() {
    echo -e "${CYAN}"
    echo "═══════════════════════════════════════════════════════════════"
    echo "🎯 RUNNING COMPREHENSIVE DEMO - ALL DOCUMENTS"
    echo "═══════════════════════════════════════════════════════════════"
    echo -e "${NC}"
    
    log "Starting comprehensive demo with ALL documents from all subdirectories..."
    info "This will process 16 documents across 3 document types:"
    echo "  • Passport: 4 documents"
    echo "  • ID Card Classic: 5 documents" 
    echo "  • Driver License: 7 documents"
    echo ""
    
    local start_time=$(date +%s)
    
    if python comprehensive_demo.py; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log "✅ Comprehensive demo completed successfully in ${duration}s"
    else
        error "❌ Comprehensive demo failed"
        return 1
    fi
    
    echo ""
}

# Function to run results analysis
run_analysis() {
    echo -e "${CYAN}"
    echo "═══════════════════════════════════════════════════════════════"
    echo "📊 RUNNING RESULTS ANALYSIS"
    echo "═══════════════════════════════════════════════════════════════"
    echo -e "${NC}"
    
    log "Analyzing comprehensive demo results..."
    
    if python analyze_results.py; then
        log "✅ Results analysis completed successfully"
    else
        error "❌ Results analysis failed"
        return 1
    fi
    
    echo ""
}

# Function to show final summary
show_final_summary() {
    echo -e "${PURPLE}"
    echo "═══════════════════════════════════════════════════════════════"
    echo "🎉 EXECUTION SUMMARY"
    echo "═══════════════════════════════════════════════════════════════"
    echo -e "${NC}"
    
    log "All demos and analysis completed successfully!"
    
    info "Generated Reports:"
    if ls samples/output/comprehensive_* >/dev/null 2>&1; then
        echo "  📊 Excel Reports:"
        ls samples/output/comprehensive_*.xlsx 2>/dev/null | sed 's/^/    • /' || true
        echo "  📄 JSON Reports:"
        ls samples/output/comprehensive_*.json 2>/dev/null | sed 's/^/    • /' || true
    fi
    
    echo ""
    info "Other Reports:"
    ls samples/output/*.xlsx samples/output/*.json 2>/dev/null | grep -v comprehensive | sed 's/^/  • /' || true
    
    echo ""
    info "Output Directory Contents:"
    ls -lah samples/output/ | tail -n +2 | sed 's/^/  /'
    
    echo ""
    log "🎯 Progress Collector System: FULLY OPERATIONAL"
    log "📄 Check reports in: samples/output/"
    
    echo -e "${GREEN}"
    echo "═══════════════════════════════════════════════════════════════"
    echo "✅ ALL DEMOS COMPLETED SUCCESSFULLY!"
    echo "═══════════════════════════════════════════════════════════════"
    echo -e "${NC}"
}

# Function to handle cleanup on error
cleanup_on_error() {
    error "Script interrupted or failed"
    info "Cleaning up..."
    # Add any cleanup tasks here if needed
    exit 1
}

# Main execution function
main() {
    # Set up error handling
    trap cleanup_on_error ERR INT TERM
    
    # Pre-flight checks
    check_directory
    check_venv
    check_dependencies
    setup_output
    
    # Show system information
    show_system_info
    
    # Record start time
    local script_start_time=$(date +%s)
    
    # Run all demos in sequence
    log "🚀 Starting complete demo execution sequence..."
    echo ""
    
    # 1. Simple test
    run_simple_test
    
    # 2. Mock demo
    run_mock_demo
    
    # 3. Comprehensive demo (main event)
    run_comprehensive_demo
    
    # 4. Results analysis
    run_analysis
    
    # Calculate total execution time
    local script_end_time=$(date +%s)
    local total_duration=$((script_end_time - script_start_time))
    
    # Show final summary
    show_final_summary
    
    log "🕒 Total execution time: ${total_duration} seconds"
    log "🎉 Complete demo execution finished successfully!"
}

# Script usage information
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  --simple       Run only simple test"
    echo "  --mock         Run only mock demo"
    echo "  --comprehensive Run only comprehensive demo"
    echo "  --analysis     Run only results analysis"
    echo "  --all          Run comprehensive demo + analysis (DEFAULT)"
    echo "  --full         Run all demos (simple + mock + comprehensive + analysis)"
    echo ""
    echo "Default: Run --all (comprehensive demo + analysis)"
}

# Parse command line arguments
case "${1:-}" in
    -h|--help)
        usage
        exit 0
        ;;
    --simple)
        check_directory
        check_venv
        check_dependencies
        show_system_info
        run_simple_test
        ;;
    --mock)
        check_directory
        check_venv
        check_dependencies
        setup_output
        show_system_info
        run_mock_demo
        ;;
    --comprehensive)
        check_directory
        check_venv
        check_dependencies
        setup_output
        show_system_info
        run_comprehensive_demo
        ;;
    --analysis)
        check_directory
        check_venv
        check_dependencies
        show_system_info
        run_analysis
        ;;
    --all)
        check_directory
        check_venv
        check_dependencies
        setup_output
        show_system_info
        run_comprehensive_demo
        run_analysis
        ;;
    --full)
        # Run all demos (simple + mock + comprehensive + analysis)
        main
        ;;
    "")
        # Run --all as default (comprehensive demo + analysis)
        check_directory
        check_venv
        check_dependencies
        setup_output
        show_system_info
        run_comprehensive_demo
        run_analysis
        ;;
    *)
        error "Unknown option: $1"
        usage
        exit 1
        ;;
esac