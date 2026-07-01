from icefall.stream_runtime.operator import PipelineOperator
from icefall.stream_runtime.operator_link import OperatorLink


class PipelineEngine:
    def __init__(self) -> None:
        self.operator_map: dict[str, PipelineOperator] = {}
        self.link_map: dict[tuple[str, str], OperatorLink] = {}

    def add_operator(self, name: str, operator: PipelineOperator) -> None:
        """
        Adds an operator to the pipeline engine.
        """
        if name in self.operator_map:
            raise ValueError(f"Operator with name {name} already exists.")
        self.operator_map[name] = operator

    def add_link(self, source: str, target: str) -> None:
        """
        Adds a link between two operators in the pipeline.
        """
        if source not in self.operator_map:
            raise ValueError(f"Source operator {source} does not exist.")
        if target not in self.operator_map:
            raise ValueError(f"Target operator {target} does not exist.")
        if (source, target) in self.link_map:
            raise ValueError(f"Link from {source} to {target} already exists.")
        source_op = self.operator_map[source]
        target_operator = self.operator_map[target]
        link = OperatorLink(source_op, target_operator)
        self.link_map[(source, target)] = link

    def get_operator(self, name: str) -> PipelineOperator:
        """
        Retrieves an operator by its name.
        """
        if name not in self.operator_map:
            raise ValueError(f"Operator with name {name} does not exist.")
        return self.operator_map[name]

    def execute(self) -> None:
        """
        Executes operators in topological order based on their dependencies.
        """
        # Build adjacency list and calculate in-degrees
        graph: dict[str, list[str]] = {name: [] for name in self.operator_map}
        in_degree = {name: 0 for name in self.operator_map}

        for (source, target), _ in self.link_map.items():
            graph[source].append(target)
            in_degree[target] += 1

        # Initialize queue with nodes that have no incoming edges
        queue = [name for name, degree in in_degree.items() if degree == 0]

        # Process nodes in topological order
        execution_order = []
        while queue:
            current = queue.pop(0)
            execution_order.append(current)

            # Process neighbors
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check if we have a valid topological sort
        if len(execution_order) != len(self.operator_map):
            raise ValueError("Cannot execute operators due to cyclic dependencies.")

        # Execute operators in topological order
        for name in execution_order:
            self.operator_map[name].execute()
