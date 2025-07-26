help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9_-]+:.*#' $(MAKEFILE_LIST) | sort | while read -r l; do \
        printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; \
    done

setup: # Set up the environment.
	@echo "Setting up the environment..."
	./local_setup/setup.sh

clean-log: # Clean up log files.
	@echo "Cleaning up log files..."
	@rm -f ./data/output/*

run-processing-image: # Run the Python script.
	$(MAKE) clean-log
	@echo "Running the Python script..."
	@python3 ./src/image_processing/py-ollama.py