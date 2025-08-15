from src.machines import plot_distribution
from src.ucb import run_ucb
from src.ep_greedy import run_epsilon_greedy
from src.apriori import run_apriori

def main():
    print("Running Slot Machine Distribution Plot...")
    plot_distribution()
    
    print("\nRunning UCB Algorithm...")
    run_ucb()
    
    print("\nRunning Epsilon-Greedy Algorithm...")
    run_epsilon_greedy()

    print("Running Apriori Algorithm...")
    run_apriori()

if __name__ == "__main__":
    main()
