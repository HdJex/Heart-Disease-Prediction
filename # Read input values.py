# Read input values
N = int(input())
percentage_share = list(map(int, input().split()))
M = int(input())
batches = [int(input()) for _ in range(M)]

# Calculate initial candy distribution
total_candies = sum(batches)
initial_distribution = [percentage * total_candies // 100 for percentage in percentage_share]

# Iterate through batches and update distribution
current_distribution = initial_distribution.copy()
for batch_size in batches:
    remaining_candies = batch_size
    new_distribution = current_distribution.copy()
    
    for i in range(N):
        if remaining_candies == 0:
            break
        diff = initial_distribution[i] - current_distribution[i]
        can_receive = min(remaining_candies, diff)
        new_distribution[i] += can_receive
        remaining_candies -= can_receive
    
    current_distribution = new_distribution

# Print the final distribution for each batch
for distribution in current_distribution:
    print(distribution, end=" ")
print()
