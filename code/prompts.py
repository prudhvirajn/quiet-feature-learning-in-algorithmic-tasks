import numpy as np
from utils import binary_adder, binary_multiplication, bitarr2string, digitarr2string, find_max_subarray, longest_common_subsequence, solve_activity_selection
from typing import List


def prompt_repeated_addition_normal(
    bitarray: np.ndarray,
    task_length: int = None,
    ablations: str = None,
    input_reverse: bool = False,
    output_reverse: bool = False,
    **kwargs,
):
    if not task_length:
        raise ValueError("Missing Bit Length")

    if not ablations:
        raise ValueError("Missing Ablation")

    k = len(bitarray) // (2 * task_length)
    result_str = ""

    for i in range(k):
        start = i * 2 * task_length
        end = start + 2 * task_length
        segment = bitarray[start:end]

        result_str += prompt_addition_normal(
            segment,
            task_length,
            ablations,
            input_reverse=input_reverse,
            output_reverse=output_reverse,
            **kwargs,
        )

    return result_str


def prompt_repeated_addition_cot(
    bitarray: np.ndarray,
    task_length: int = None,
    ablations: str = None,
    input_reverse: bool = False,
    output_reverse: bool = False,
    skip_line: int = 1,
):
    if not task_length:
        raise ValueError("Missing Bit Length")

    if not ablations:
        raise ValueError("Missing Ablation")

    k = bitarray.size // (2 * task_length)
    result_str = ""

    for i in range(k):
        start = i * 2 * task_length
        end = start + 2 * task_length
        segment = bitarray[start:end]

        result_str += prompt_addition_cot(
            segment,
            task_length,
            ablations,
            input_reverse=input_reverse,
            output_reverse=output_reverse,
            skip_line=skip_line,
        )

    return result_str


def prompt_multiplication_normal(
    bitarray: np.ndarray,
    task_length: int = None,
    input_reverse: bool = False,
    output_reverse: bool = False,
    **kwargs,
) -> str:
    if not task_length:
        raise ValueError("Missing Bit Length")

    num_bits = task_length

    op1 = np.array(bitarray[:num_bits], dtype=np.uint8)
    op2 = np.array(bitarray[num_bits:], dtype=np.uint8)

    result = binary_multiplication(op1, op2)

    if input_reverse:
        op1 = op1[::-1]
        op2 = op2[::-1]

    if output_reverse:
        result = result[::-1]

    return f"{bitarr2string(op1)}+{bitarr2string(op2)}={bitarr2string(result)}<EOS>"


def prompt_parity_normal(
    bitarray: np.ndarray,
    **kwargs,
) -> str:
    result = np.sum(bitarray) % 2

    return f"{bitarr2string(bitarray)}={result}<EOS>"

def prompt_majority_normal(
    bitarray: np.ndarray,
    **kwargs,
) -> str:
    result = 1 if np.sum(bitarray) > len(bitarray) // 2 else 0

    return f"{bitarr2string(bitarray)}={result}<EOS>"

def prompt_majority_of_majority_normal(
    bitarray: np.ndarray,
    num_groups: int = 4,
    **kwargs,
) -> str:
    # Split the bitarray into groups
    group_size = len(bitarray) // num_groups
    groups = np.array_split(bitarray, num_groups)
    
    # Calculate majority for each group
    group_majorities = [1 if np.sum(group) > len(group) // 2 else 0 for group in groups]
    
    # Calculate the final majority of the group majorities
    final_result = 1 if np.sum(group_majorities) > num_groups // 2 else 0

    return f"{bitarr2string(bitarray)}={final_result}<EOS>"


def prompt_sorting_normal(
    digitarray: np.ndarray,
    task_length: int = None,
    **kwargs,
) -> str:
    digitstring = digitarray[0] # [2:]
    
    return f"{digitstring}={','.join(sorted(digitstring.split(',')))}<EOS>".upper()

def prompt_inner_product_mod2_parity(
    bitarray: np.ndarray,
    task_length: int = None,
    input_reverse: bool = False,
    output_reverse: bool = False,
    **kwargs,
) -> str:
    if not task_length:
        raise ValueError("Missing Bit Length")

    num_bits = task_length

    op1 = bitarray[:num_bits]
    op2 = bitarray[num_bits:]

    result = np.dot(op1, op2) % 2

    # if input_reverse:
    #     op1 = op1[::-1]
    #     op2 = op2[::-1]

    # if output_reverse:
    #     result = result[::-1]

    return f"{bitarr2string(op1)}+{bitarr2string(op2)}={result}<EOS>"

def prompt_binary_sorting_normal(
    hexarray: np.ndarray,
    task_length: int = None,
    **kwargs,
) -> str:
    hexstring = hexarray[0][2:]
    sorted_hexstring = ''.join(sorted(hexstring))
    
    binary_string = bin(int(hexstring, 16))[2:].zfill(4 * task_length)
    output = bin(int(sorted_hexstring, 16))[2:].zfill(4 * task_length)
    
    return f"{binary_string}={output}<EOS>".upper()


def prompt_maximum_subarray_normal(
    input_array: np.ndarray,
    task_length: int = None,
    **kwargs,
) -> str:
    subarray, max_sum = find_max_subarray(input_array)

    input_array += 9
    subarray += 9

    max_tokens = 2 * task_length + 2

    if max(input_array) > 18:
        raise ValueError("Max input array value is greater than 18")
    
    if max(subarray) > 18:
        raise ValueError("Max subarray value is greater than 18")
    
    input_array_string = "".join([chr(val + 65) for val in input_array])
    subarray_string = "".join([chr(val + 65) for val in subarray])

    token_string = f"{input_array_string}={subarray_string}"
    end_of_token_string = "<EOS>" * (max_tokens - len(token_string))

    return f"{token_string}{end_of_token_string}"

def prompt_activity_selection_normal(
    input_array: np.ndarray,
    task_length: int = None,
    **kwargs,
) -> str:

    event_start_times = input_array[:task_length]
    input_durations = input_array[task_length:]

    event_end_times = event_start_times + input_durations
    input_array = np.concatenate((event_start_times, event_end_times))
    
    selected_activities = solve_activity_selection(event_start_times, event_end_times)

    max_tokens = 4 * task_length + 2 
    
    input_array_string = "".join([chr(val + 65) for val in input_array])
    subarray_string = "".join([chr(val + 65) for val in selected_activities])

    token_string = f"{input_array_string}={subarray_string}"
    end_of_token_string = "<EOS>" * (max_tokens - len(token_string))

    return f"{token_string}{end_of_token_string}"


def prompt_longest_common_subsquence(
    input_array: np.ndarray,
    task_length: int = None,
    **kwargs,
) -> str:

    string1 = input_array[:task_length]
    string2 = input_array[task_length:]

    lcs_arr = longest_common_subsequence(string1, string2)

    max_tokens = 3 * task_length + 2 
    
    input_array_string = "".join([chr(val + 65) for val in input_array])
    lcs_string = "".join([chr(val + 65) for val in lcs_arr])

    token_string = f"{input_array_string}={lcs_string}"
    end_of_token_string = "<EOS>" * (max_tokens - len(token_string))

    return f"{token_string}{end_of_token_string}"



def prompt_pathfinding_normal(
    edge_list: np.ndarray,
    number_of_nodes: int = None,
    task_length: int = None,
    **kwargs,
) -> str:
    from sage.all import Graph # type: ignore

    if not number_of_nodes:
        raise ValueError("Missing Number of Nodes")

    G = Graph()

    nodes = list(range(number_of_nodes))
    G.add_vertices(nodes)

    edge_string = "".join([chr(node + 65) for node in edge_list])
    edge_list = edge_list.reshape(-1, 2)

    G.add_edges(edge_list)
    
    max_tokens = number_of_nodes * (number_of_nodes - 1) + 3 + number_of_nodes + 1 

    selected_nodes = np.random.choice(nodes, size=2, replace=False)
    path = G.shortest_path(selected_nodes[0], selected_nodes[1])

    path_string = "".join([chr(node + 65) for node in path])
    example_string = f"{chr(selected_nodes[0] + 65)}{chr(selected_nodes[1] + 65)}{edge_string}={path_string}"
    end_of_token_string = "<EOS>" * (max_tokens - len(example_string))
    
    return f"{example_string}{end_of_token_string}"

def prompt_bfs_normal(
    edge_list: np.ndarray,
    number_of_nodes: int = None,
    task_length: int = None,
    **kwargs,
) -> str:
    from sage.all import Graph # type: ignore

    if not number_of_nodes:
        raise ValueError("Missing Number of Nodes")

    G = Graph()

    nodes = list(range(number_of_nodes))
    G.add_vertices(nodes)

    edge_string = "".join([chr(node + 65) for node in edge_list])
    edge_list = edge_list.reshape(-1, 2)

    G.add_edges(edge_list)
    
    max_tokens = number_of_nodes * (number_of_nodes - 1) + 2 + number_of_nodes + 1 

    selected_nodes = np.random.choice(nodes, size=1, replace=False)
    visited_nodes = list(G.breadth_first_search(selected_nodes[0]))

    visited_nodes_string = "".join([chr(node + 65) for node in visited_nodes])
    example_string = f"{chr(selected_nodes[0] + 65)}{edge_string}={visited_nodes_string}"
    end_of_token_string = "<EOS>" * (max_tokens - len(example_string))
    
    return f"{example_string}{end_of_token_string}"

def prompt_dfs_normal(
    edge_list: np.ndarray,
    number_of_nodes: int = None,
    task_length: int = None,
    **kwargs,
) -> str:
    from sage.all import Graph # type: ignore

    if not number_of_nodes:
        raise ValueError("Missing Number of Nodes")

    G = Graph()

    nodes = list(range(number_of_nodes))
    G.add_vertices(nodes)

    edge_string = "".join([chr(node + 65) for node in edge_list])
    edge_list = edge_list.reshape(-1, 2)

    G.add_edges(edge_list)
    
    max_tokens = number_of_nodes * (number_of_nodes - 1) + 2 + number_of_nodes + 1 

    selected_nodes = np.random.choice(nodes, size=1, replace=False)
    visited_nodes = list(G.depth_first_search(selected_nodes[0]))

    visited_nodes_string = "".join([chr(node + 65) for node in visited_nodes])
    example_string = f"{chr(selected_nodes[0] + 65)}{edge_string}={visited_nodes_string}"
    end_of_token_string = "<EOS>" * (max_tokens - len(example_string))
    
    return f"{example_string}{end_of_token_string}"

def prompt_topological_sorting_normal(
    edge_list: np.ndarray,
    number_of_nodes: int = None,
    task_length: int = None,
    **kwargs,
) -> str:
    import numpy as np
    from sage.all import DiGraph  # type: ignore

    if number_of_nodes is None:
        raise ValueError("Missing Number of Nodes")

    # Create a list of nodes and sample a random permutation (this will guide our edge orientation)
    nodes = list(range(number_of_nodes))
    random_order = list(np.random.permutation(nodes))
    order_index = {node: idx for idx, node in enumerate(random_order)}

    # Reshape the edge_list into pairs and orient each edge according to the random order.
    # For an edge (u, v), if u comes before v in random_order, direct it u -> v; otherwise, v -> u.
    edge_list = edge_list.reshape(-1, 2)
    oriented_edges = []
    for u, v in edge_list:
        if order_index[u] < order_index[v]:
            oriented_edges.append((u, v))
        else:
            oriented_edges.append((v, u))

    # Build a string representation of the oriented edges, converting nodes to letters (A, B, C, ...)
    oriented_edge_string = "".join([chr(u + 65) + chr(v + 65) for u, v in oriented_edges])

    # Initialize a Sage DiGraph, add vertices and the oriented edges.
    G = DiGraph()
    G.add_vertices(nodes)
    G.add_edges(oriented_edges)

    # Obtain a topological order using Sage's topological_sort() method.
    # Note that if the DAG has multiple valid orders, this method will return one valid order.
    topo_sort = G.topological_sort()
    topo_sort_string = "".join([chr(node + 65) for node in topo_sort])

    # Define a maximum token length as in your BFS example.
    max_tokens = number_of_nodes * (number_of_nodes - 1) + 2 + number_of_nodes + 1

    # Construct the output string: using the first node (as a letter) followed by the oriented edges, an "=" sign,
    # and finally the topological ordering.
    example_string = f"{oriented_edge_string}={topo_sort_string}"
    end_of_token_string = "<EOS>" * (max_tokens - len(example_string))

    return f"{example_string}{end_of_token_string}"

def prompt_min_spanning_tree_kruskal_normal(
    edge_list: np.ndarray,
    number_of_nodes: int = None,
    task_length: int = None,
    **kwargs,
) -> str:
    from sage.all import Graph # type: ignore
    from sage.graphs.spanning_tree import kruskal # type: ignore

    if not number_of_nodes:
        raise ValueError("Missing Number of Nodes")

    G = Graph()

    nodes = list(range(number_of_nodes))
    G.add_vertices(nodes)

    edge_string = "".join([chr(node + 65) for node in edge_list])
    edge_list = edge_list.reshape(-1, 2)

    G.add_edges(edge_list)
    
    max_tokens = number_of_nodes * (number_of_nodes - 1) + 2 + 2 * number_of_nodes - 1

    min_spanning_tree_edges = kruskal(G)
    min_spanning_tree_edge_list = [item for tup in min_spanning_tree_edges for item in tup[:2]]
    min_spanning_tree_edges_string = "".join([chr(node + 65) for node in min_spanning_tree_edge_list])

    example_string = f"{edge_string}={min_spanning_tree_edges_string}"
    end_of_token_string = "<EOS>" * (max_tokens - len(example_string))
    
    return f"{example_string}{end_of_token_string}"

def prompt_longest_path_normal(
    edge_list: np.ndarray,
    number_of_nodes: int = None,
    task_length: int = None,
    **kwargs,
) -> str:
    from sage.all import Graph # type: ignore

    if not number_of_nodes:
        raise ValueError("Missing Number of Nodes")

    G = Graph()

    nodes = list(range(number_of_nodes))
    G.add_vertices(nodes)

    edge_string = "".join([chr(node + 65) for node in edge_list])
    edge_list = edge_list.reshape(-1, 2)

    G.add_edges(edge_list.reshape(-1, 2))

    path = G.longest_path()
    
    max_tokens = number_of_nodes * (number_of_nodes - 1) + 1 + number_of_nodes + 1 

    path_string = "".join([chr(node + 65) for node in path])
    example_string = f"{edge_string}={path_string}"
    end_of_token_string = "<EOS>" * (max_tokens - len(example_string))
    return f"{example_string}{end_of_token_string}"

def prompt_eulerian_circuit_normal(
    edge_list: np.ndarray,
    number_of_nodes: int = None,
    task_length: int = None,
    **kwargs,
) -> str:
    from sage.all import Graph # type: ignore

    if not number_of_nodes:
        raise ValueError("Missing Number of Nodes")

    G = Graph()

    nodes = list(range(number_of_nodes))
    G.add_vertices(nodes)

    edge_string = "".join([chr(node + 65) for node in edge_list])
    edge_list = edge_list.reshape(-1, 2)

    G.add_edges(edge_list.reshape(-1, 2))

    edges, vertices = G.eulerian_circuit(return_vertices=True)
    
    max_tokens = number_of_nodes * (number_of_nodes - 1) + 1 + (number_of_nodes * (number_of_nodes - 1)) // 2 + 1 

    vertex_string = "".join([chr(node + 65) for node in vertices])
    example_string = f"{edge_string}={vertex_string}"
    end_of_token_string = "<EOS>" * (max_tokens - len(example_string))
    return f"{example_string}{end_of_token_string}"

def prompt_max_independent_set_normal(
    edge_list: np.ndarray,
    number_of_nodes: int = None,
    task_length: int = None,
    **kwargs,
) -> str:
    from sage.all import Graph # type: ignore
    from sage.graphs.independent_sets import IndependentSets # type: ignore

    if not number_of_nodes:
        raise ValueError("Missing Number of Nodes")

    G = Graph()

    nodes = list(range(number_of_nodes))
    G.add_vertices(nodes)

    edge_string = "".join([chr(node + 65) for node in edge_list])
    edge_list = edge_list.reshape(-1, 2)

    G.add_edges(edge_list.reshape(-1, 2))
    
    Im = list(IndependentSets(G, maximal=True))
    
    maximum_independent_set = max(Im, key=len)
    
    max_tokens = number_of_nodes * (number_of_nodes - 1) + 1 + number_of_nodes + 1 

    vertex_string = "".join([chr(node + 65) for node in maximum_independent_set])
    
    example_string = f"{edge_string}={vertex_string}"
    end_of_token_string = "<EOS>" * (max_tokens - len(example_string))
    return f"{example_string}{end_of_token_string}"

def prompt_maxcut_normal(
    edge_list: np.ndarray,
    number_of_nodes: int = None,
    task_length: int = None,
    **kwargs,
) -> str:
    from sage.all import Graph # type: ignore

    if not number_of_nodes:
        raise ValueError("Missing Number of Nodes")

    G = Graph()

    nodes = list(range(number_of_nodes))
    G.add_vertices(nodes)

    edge_string = "".join([chr(node + 65) for node in edge_list])
    edge_list = edge_list.reshape(-1, 2)

    G.add_edges(edge_list.reshape(-1, 2))

    [ value, edges, [ setA, setB ]] = G.max_cut(vertices=True)
    
    max_tokens = number_of_nodes * (number_of_nodes - 1) + 1 + number_of_nodes + 2 

    path_string = "".join([chr(node + 65) for node in setA])
    path_string += "|"
    path_string += "".join([chr(node + 65) for node in setB])
    
    example_string = f"{edge_string}={path_string}"
    end_of_token_string = "<EOS>" * (max_tokens - len(example_string))
    return f"{example_string}{end_of_token_string}"

def prompt_addition_normal(
    bitarray: np.ndarray,
    task_length: int = None,
    input_reverse: bool = False,
    output_reverse: bool = False,
    **kwargs,
) -> str:
    if not task_length:
        raise ValueError("Missing Bit Length")

    num_bits = task_length

    op1 = bitarray[:num_bits]
    op2 = bitarray[num_bits:]

    result = binary_adder(op1, op2)

    if input_reverse:
        op1 = op1[::-1]
        op2 = op2[::-1]

    if output_reverse:
        result = result[::-1]

    return f"{bitarr2string(op1)}+{bitarr2string(op2)}={bitarr2string(result)}<EOS>"


def prompt_addition_cot(
    bitarray: np.ndarray,
    task_length: int = None,
    ablations: str = None,
    input_reverse: bool = False,
    output_reverse: bool = False,
    skip_line: int = 1,
) -> str:
    if not task_length:
        raise ValueError("Missing Bit Length")

    if not ablations:
        raise ValueError("Missing Ablation")

    num_bits = task_length

    op1 = bitarray[:num_bits]
    op2 = bitarray[num_bits:]

    lines = []

    # Initialize variables to store the sum and carry
    sum_array = np.zeros(num_bits + 1, dtype=np.uint8)
    carry = 0
    prev_carry = 0

    # Iterate through the arrays and perform binary addition
    for i in range(num_bits - 1, -1, -1):
        a = op1[i]
        b = op2[i]
        prev_carry = carry

        # Calculate the sum bit
        sum_bit = a ^ b ^ carry

        # Calculate the carry bit
        carry = (a & b) | (carry & (a ^ b))

        # Store the sum bit in the result array
        sum_array[i + 1] = sum_bit

        if i % skip_line == 0:
            variables = [a, b, prev_carry, sum_bit, carry]
            included_vars = [
                str(var) for var, bit in zip(variables, ablations) if bit == "1"
            ]

            # Join the included variables with formatting
            line = "".join(included_vars)
            lines.append(line)

    # If there's a final carry after the loop, append it as the most significant bit
    if carry:
        sum_array[0] = carry

    if input_reverse:
        op1 = op1[::-1]
        op2 = op2[::-1]

    lines = [f"{bitarr2string(op1)}+{bitarr2string(op2)}="] + lines

    if output_reverse:
        sum_array = sum_array[::-1]

    lines.append(f"={bitarr2string(sum_array)}<EOS>")

    return "".join(lines)


def prompt_multiplication_cot(
    bitarray: np.ndarray,
    task_length: int,
    reverse: bool = False,
    cot_type: str = "f",
    skip_lines: int = 1,
    newline_tokens=True,
) -> str:
    raise Exception("Not implemented")
    num_bits = task_length

    op1 = bitarray[:num_bits]
    op2 = bitarray[num_bits:]

    lines = [f"{bitarr2string(op1)}+{bitarr2string(op2)}="]

    # Initialize variables to store partial products and their sum
    partial_products = []
    sum_array = np.zeros(2 * num_bits, dtype=np.uint8)

    # Compute partial products
    for i in range(num_bits):
        if op2[num_bits - 1 - i] == 1:
            partial_product = np.pad(
                op1, (num_bits - i, i), mode="constant", constant_values=0
            )
        else:
            partial_product = np.zeros(2 * num_bits, dtype=np.uint8)

        partial_products.append(partial_product)
        lines.append(bitarr2string(partial_product))

    partial_sum = bitarr2string(partial_products[0])

    for i in range(1, num_bits):
        partial_sum = int(partial_sum, 2) + int(bitarr2string(partial_products[i]), 2)
        partial_sum = np.binary_repr(partial_sum, 2 * num_bits)

        lines.append(partial_sum)

    sum_array = lines[-1]

    if reverse:
        sum_array = sum_array[::-1]

    lines.append(f"={sum_array}<EOS>")

    return "\n".join(lines) if newline_tokens else "".join(lines)
