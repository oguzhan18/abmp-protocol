"""
ABMP Bio-Inspired Routing Engine
Staff Engineer Level: Clean Architecture with Strategy Pattern
"""

import time
import random
import math
import threading
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.interfaces import (
    IRoutingEngine, IRoutingStrategy, RoutingDecision, 
    NodeMetrics, NodeID, NetworkState, IEventPublisher
)
from common.exceptions import (
    RoutingError, NodeNotFoundError, RouteNotFoundError, 
    InvalidConfigurationError, handle_abmp_exception
)
from common.events import EventType, publish_event


logger = logging.getLogger(__name__)


class AntType(Enum):
    """Digital ant types for different routing behaviors"""
    FORAGER = "forager"      # Explores new paths
    WORKER = "worker"        # Carries data traffic
    SCOUT = "scout"          # Monitors network state
    GUARD = "guard"          # Security-focused routing
    REPAIR = "repair"        # Healing-focused routing


@dataclass(frozen=True)
class PheromoneTrail:
    """Immutable pheromone trail for bio-routing"""
    source: NodeID
    destination: NodeID
    intensity: float
    quality_score: float
    age: float
    trail_type: str
    last_reinforcement: float = field(default_factory=time.time)
    usage_count: int = 0
    
    def __post_init__(self):
        if not (0.0 <= self.intensity <= 1.0):
            raise InvalidConfigurationError("intensity", self.intensity, "0.0-1.0")
        if not (0.0 <= self.quality_score <= 1.0):
            raise InvalidConfigurationError("quality_score", self.quality_score, "0.0-1.0")
    
    def evaporate(self, evaporation_rate: float) -> 'PheromoneTrail':
        """Create new trail with evaporation applied"""
        time_elapsed = time.time() - self.last_reinforcement
        decay_factor = math.exp(-evaporation_rate * time_elapsed)
        new_intensity = self.intensity * decay_factor
        
        return PheromoneTrail(
            source=self.source,
            destination=self.destination,
            intensity=new_intensity,
            quality_score=self.quality_score,
            age=self.age + time_elapsed,
            trail_type=self.trail_type,
            last_reinforcement=self.last_reinforcement,
            usage_count=self.usage_count
        )
    
    def reinforce(self, amount: float) -> 'PheromoneTrail':
        """Create new trail with reinforcement applied"""
        new_intensity = min(1.0, self.intensity + amount)
        
        return PheromoneTrail(
            source=self.source,
            destination=self.destination,
            intensity=new_intensity,
            quality_score=self.quality_score,
            age=self.age,
            trail_type=self.trail_type,
            last_reinforcement=time.time(),
            usage_count=self.usage_count + 1
        )


@dataclass
class DigitalAnt:
    """Digital ant for path exploration"""
    ant_id: str
    ant_type: AntType
    source_node: NodeID
    destination_node: NodeID
    current_node: NodeID
    path_history: List[NodeID] = field(default_factory=list)
    energy: float = 100.0
    success_rate: float = 0.0
    creation_time: float = field(default_factory=time.time)
    max_hops: int = 50
    
    @property
    def age(self) -> float:
        """Get ant age in seconds"""
        return time.time() - self.creation_time
    
    @property
    def is_alive(self) -> bool:
        """Check if ant is still alive"""
        return self.energy > 0 and len(self.path_history) < self.max_hops
    
    def consume_energy(self, amount: float = 1.0) -> None:
        """Consume ant energy"""
        self.energy = max(0, self.energy - amount)
    
    def calculate_fitness(self) -> float:
        """Calculate ant fitness score"""
        if not self.path_history:
            return 0.0
        
        path_length_penalty = len(self.path_history) / self.max_hops
        energy_bonus = self.energy / 100.0
        success_bonus = self.success_rate
        
        return success_bonus * 0.6 + energy_bonus * 0.2 + (1 - path_length_penalty) * 0.2


class AntColonyStrategy(IRoutingStrategy):
    """
    Ant Colony Optimization routing strategy
    Implements Strategy Pattern for pluggable routing algorithms
    """
    
    def __init__(self, 
                 alpha: float = 1.0,      # Pheromone importance
                 beta: float = 2.0,       # Heuristic importance
                 gamma: float = 0.5,      # Swarm intelligence factor
                 evaporation_rate: float = 0.02):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.evaporation_rate = evaporation_rate
        self.name = "AntColonyOptimization"
        
        # Pheromone management
        self._pheromone_trails: Dict[Tuple[NodeID, NodeID], PheromoneTrail] = {}
        self._active_ants: Dict[str, DigitalAnt] = {}
        self._ant_counter = 0
        self._trail_lock = threading.RLock()
        
        logger.info(f"Ant Colony Strategy initialized: α={alpha}, β={beta}, γ={gamma}")
    
    def calculate_route(self, source: NodeID, destination: NodeID, 
                       network_state: NetworkState) -> RoutingDecision:
        """Calculate optimal route using ant colony optimization"""
        try:
            # Get available neighbors
            neighbors = self._get_neighbors(source, network_state)
            if not neighbors:
                return RoutingDecision(
                    next_hop=None,
                    confidence=0.0,
                    alternative_hops=[],
                    reasoning="No neighbors available"
                )
            
            # Direct connection check
            if destination in neighbors:
                return RoutingDecision(
                    next_hop=destination,
                    confidence=1.0,
                    alternative_hops=[],
                    reasoning="Direct connection to destination"
                )
            
            # Deploy scout ant for route discovery
            scout_ant = self._create_ant(AntType.SCOUT, source, destination)
            path = self._simulate_ant_journey(scout_ant, network_state)
            
            if len(path) < 2:
                return RoutingDecision(
                    next_hop=None,
                    confidence=0.0,
                    alternative_hops=[],
                    reasoning="No path found by scout ant"
                )
            
            # Calculate confidence based on pheromone trails and path quality
            next_hop = path[1]
            confidence = self._calculate_path_confidence(path)
            alternatives = self._find_alternative_hops(source, destination, next_hop, neighbors)
            
            # Update pheromone trails
            self._update_pheromone_trails(path, scout_ant.calculate_fitness())
            
            return RoutingDecision(
                next_hop=next_hop,
                confidence=confidence,
                alternative_hops=alternatives,
                reasoning=f"ACO path via {len(path)-1} hops, fitness={scout_ant.calculate_fitness():.3f}"
            )
            
        except Exception as e:
            logger.error(f"ACO routing calculation failed: {e}")
            return RoutingDecision(
                next_hop=None,
                confidence=0.0,
                alternative_hops=[],
                reasoning=f"Routing failed: {str(e)}"
            )
    
    def get_algorithm_name(self) -> str:
        """Get algorithm name"""
        return self.name
    
    def _create_ant(self, ant_type: AntType, source: NodeID, destination: NodeID) -> DigitalAnt:
        """Create a new digital ant"""
        self._ant_counter += 1
        ant_id = f"{ant_type.value}_{self._ant_counter}_{int(time.time() * 1000)}"
        
        # Ant-specific parameters
        max_hops = 30 if ant_type == AntType.FORAGER else 20
        initial_energy = 80.0 if ant_type == AntType.SCOUT else 100.0
        
        ant = DigitalAnt(
            ant_id=ant_id,
            ant_type=ant_type,
            source_node=source,
            destination_node=destination,
            current_node=source,
            max_hops=max_hops,
            energy=initial_energy
        )
        
        self._active_ants[ant_id] = ant
        return ant
    
    def _simulate_ant_journey(self, ant: DigitalAnt, network_state: NetworkState) -> List[NodeID]:
        """Simulate ant's journey through network"""
        path = [ant.current_node]
        
        while ant.is_alive and ant.current_node != ant.destination_node:
            # Get available neighbors
            neighbors = self._get_neighbors(ant.current_node, network_state)
            
            # Remove visited nodes to avoid cycles
            available_neighbors = [n for n in neighbors if n not in ant.path_history]
            
            if not available_neighbors:
                # Dead end - allow revisiting with penalty
                available_neighbors = neighbors
                ant.consume_energy(5.0)  # Penalty for backtracking
            
            # Select next node using ACO probability
            next_node = self._select_next_node(ant, available_neighbors, network_state)
            
            if next_node is None:
                break
            
            # Move ant
            ant.path_history.append(ant.current_node)
            ant.current_node = next_node
            ant.consume_energy(1.0)
            path.append(next_node)
            
            # Update pheromone trail
            self._leave_pheromone(ant.path_history[-1] if ant.path_history else ant.source_node,
                                next_node, ant)
        
        # Mark success if destination reached
        if ant.current_node == ant.destination_node:
            ant.success_rate = 1.0
            self._reinforce_successful_path(path, ant)
        
        return path
    
    def _select_next_node(self, ant: DigitalAnt, neighbors: List[NodeID], 
                         network_state: NetworkState) -> Optional[NodeID]:
        """Select next node using ACO probability calculation"""
        if not neighbors:
            return None
        
        # Calculate selection probabilities
        probabilities = {}
        total_probability = 0.0
        
        for neighbor in neighbors:
            # Pheromone level
            pheromone_intensity = self._get_pheromone_intensity(ant.current_node, neighbor)
            
            # Heuristic information (node quality, distance estimate)
            heuristic = self._calculate_heuristic(neighbor, ant.destination_node, network_state)
            
            # Swarm intelligence factor
            swarm_factor = self._get_swarm_preference(ant.current_node, neighbor)
            
            # Ant-specific behavior modifier
            behavior_factor = self._get_ant_behavior_factor(ant, neighbor, network_state)
            
            # Combined probability
            probability = (
                (pheromone_intensity ** self.alpha) *
                (heuristic ** self.beta) *
                (swarm_factor ** self.gamma) *
                behavior_factor
            )
            
            probabilities[neighbor] = probability
            total_probability += probability
        
        # Normalize probabilities
        if total_probability > 0:
            for node in probabilities:
                probabilities[node] /= total_probability
        
        # Roulette wheel selection
        return self._roulette_wheel_selection(probabilities)
    
    def _get_pheromone_intensity(self, from_node: NodeID, to_node: NodeID) -> float:
        """Get pheromone intensity between nodes"""
        with self._trail_lock:
            trail_key = (from_node, to_node)
            if trail_key in self._pheromone_trails:
                trail = self._pheromone_trails[trail_key]
                evaporated_trail = trail.evaporate(self.evaporation_rate)
                self._pheromone_trails[trail_key] = evaporated_trail
                return evaporated_trail.intensity
            return 0.1  # Minimum pheromone level
    
    def _calculate_heuristic(self, node: NodeID, destination: NodeID, 
                           network_state: NetworkState) -> float:
        """Calculate heuristic value (desirability) of node"""
        # Simple distance-based heuristic (can be enhanced with topology info)
        distance_estimate = abs(node - destination)
        distance_factor = 1.0 / (1.0 + distance_estimate * 0.1)
        
        # Node quality from network state
        node_quality = network_state.get('node_quality', {}).get(node, 0.5)
        
        return distance_factor * 0.6 + node_quality * 0.4
    
    def _get_swarm_preference(self, from_node: NodeID, to_node: NodeID) -> float:
        """Get swarm collective preference for this route"""
        # Simplified swarm intelligence - can be enhanced with machine learning
        route_key = f"route_{from_node}_{to_node}"
        # TODO: Implement collective memory and emergent behavior
        return 0.5  # Neutral preference for now
    
    def _get_ant_behavior_factor(self, ant: DigitalAnt, neighbor: NodeID, 
                                network_state: NetworkState) -> float:
        """Get ant-type specific behavior factor"""
        base_factor = 1.0
        
        if ant.ant_type == AntType.FORAGER:
            # Forager ants prefer exploration
            base_factor = 0.8 + random.random() * 0.4
        elif ant.ant_type == AntType.SCOUT:
            # Scout ants prefer fast paths
            base_factor = 1.2
        elif ant.ant_type == AntType.GUARD:
            # Guard ants prefer secure nodes
            security_score = network_state.get('security_scores', {}).get(neighbor, 0.5)
            base_factor = security_score * 1.5
        elif ant.ant_type == AntType.REPAIR:
            # Repair ants prefer problematic paths for healing
            problem_score = network_state.get('problem_scores', {}).get(neighbor, 0.1)
            base_factor = 1.0 + problem_score
        
        return base_factor
    
    def _roulette_wheel_selection(self, probabilities: Dict[NodeID, float]) -> Optional[NodeID]:
        """Roulette wheel selection for next node"""
        if not probabilities:
            return None
        
        rand_val = random.random()
        cumulative_prob = 0.0
        
        for node, prob in probabilities.items():
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                return node
        
        # Fallback: return highest probability node
        return max(probabilities, key=probabilities.get)
    
    def _leave_pheromone(self, from_node: NodeID, to_node: NodeID, ant: DigitalAnt) -> None:
        """Leave pheromone trail on path segment"""
        with self._trail_lock:
            trail_key = (from_node, to_node)
            
            # Calculate pheromone amount based on ant fitness
            pheromone_amount = ant.calculate_fitness() * 0.1
            
            if trail_key in self._pheromone_trails:
                # Reinforce existing trail
                existing_trail = self._pheromone_trails[trail_key]
                self._pheromone_trails[trail_key] = existing_trail.reinforce(pheromone_amount)
            else:
                # Create new trail
                self._pheromone_trails[trail_key] = PheromoneTrail(
                    source=from_node,
                    destination=to_node,
                    intensity=pheromone_amount,
                    quality_score=ant.calculate_fitness(),
                    age=0.0,
                    trail_type=ant.ant_type.value
                )
    
    def _reinforce_successful_path(self, path: List[NodeID], ant: DigitalAnt) -> None:
        """Reinforce pheromone trails for successful path"""
        if len(path) < 2:
            return
        
        # Global reinforcement for entire path
        path_quality = ant.calculate_fitness()
        reinforcement_amount = path_quality / len(path)  # Distributed reinforcement
        
        with self._trail_lock:
            for i in range(len(path) - 1):
                from_node, to_node = path[i], path[i + 1]
                trail_key = (from_node, to_node)
                
                if trail_key in self._pheromone_trails:
                    existing_trail = self._pheromone_trails[trail_key]
                    self._pheromone_trails[trail_key] = existing_trail.reinforce(reinforcement_amount)
    
    def _get_neighbors(self, node: NodeID, network_state: NetworkState) -> List[NodeID]:
        """Get neighbors of a node from network state"""
        topology = network_state.get('topology', {})
        return topology.get(node, [])
    
    def _calculate_path_confidence(self, path: List[NodeID]) -> float:
        """Calculate confidence score for path"""
        if len(path) < 2:
            return 0.0
        
        total_confidence = 0.0
        
        for i in range(len(path) - 1):
            from_node, to_node = path[i], path[i + 1]
            pheromone_intensity = self._get_pheromone_intensity(from_node, to_node)
            total_confidence += pheromone_intensity
        
        # Average confidence with path length penalty
        avg_confidence = total_confidence / (len(path) - 1)
        path_length_factor = max(0.1, 1.0 - (len(path) - 2) * 0.1)
        
        return avg_confidence * path_length_factor
    
    def _find_alternative_hops(self, source: NodeID, destination: NodeID, 
                              primary_hop: NodeID, neighbors: List[NodeID]) -> List[NodeID]:
        """Find alternative next hops"""
        alternatives = []
        
        for neighbor in neighbors:
            if neighbor != primary_hop:
                # Score alternative based on pheromone and heuristic
                pheromone = self._get_pheromone_intensity(source, neighbor)
                heuristic = abs(neighbor - destination)  # Simple distance
                score = pheromone * 0.7 + (1.0 / (1.0 + heuristic)) * 0.3
                alternatives.append((neighbor, score))
        
        # Sort by score and return top alternatives
        alternatives.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in alternatives[:3]]
    
    def _update_pheromone_trails(self, path: List[NodeID], fitness: float) -> None:
        """Update pheromone trails based on path quality"""
        # This method provides additional global updating beyond local trail leaving
        if len(path) < 2:
            return
        
        update_amount = fitness * 0.05  # Global update factor
        
        with self._trail_lock:
            for i in range(len(path) - 1):
                from_node, to_node = path[i], path[i + 1]
                trail_key = (from_node, to_node)
                
                if trail_key in self._pheromone_trails:
                    existing_trail = self._pheromone_trails[trail_key]
                    self._pheromone_trails[trail_key] = existing_trail.reinforce(update_amount)
    
    def cleanup_expired_trails(self) -> None:
        """Remove expired or weak pheromone trails"""
        with self._trail_lock:
            current_time = time.time()
            expired_trails = []
            
            for trail_key, trail in self._pheromone_trails.items():
                # Remove trails older than 5 minutes with low intensity
                if (current_time - trail.last_reinforcement > 300 and 
                    trail.intensity < 0.05):
                    expired_trails.append(trail_key)
            
            for trail_key in expired_trails:
                del self._pheromone_trails[trail_key]
            
            if expired_trails:
                logger.debug(f"Cleaned up {len(expired_trails)} expired pheromone trails")
    
    def get_statistics(self) -> Dict[str, any]:
        """Get routing strategy statistics"""
        with self._trail_lock:
            return {
                'algorithm': self.name,
                'active_trails': len(self._pheromone_trails),
                'active_ants': len(self._active_ants),
                'parameters': {
                    'alpha': self.alpha,
                    'beta': self.beta,
                    'gamma': self.gamma,
                    'evaporation_rate': self.evaporation_rate
                }
            }


class BiologicalRoutingEngine(IRoutingEngine):
    """
    Main biological routing engine
    Implements Context pattern for strategy management
    """
    
    def __init__(self, node_id: NodeID, strategy: IRoutingStrategy = None):
        self.node_id = node_id
        self._strategy = strategy or AntColonyStrategy()
        self._routing_table: Dict[NodeID, List[Tuple[NodeID, float]]] = {}
        self._network_state: NetworkState = {
            'topology': {},
            'node_quality': {},
            'security_scores': {},
            'problem_scores': {}
        }
        self._lock = threading.RLock()
        
        logger.info(f"Biological routing engine initialized for Node {node_id}")
    
    @handle_abmp_exception
    def find_route(self, destination: NodeID, current_metrics: NodeMetrics) -> RoutingDecision:
        """Find optimal route to destination"""
        if destination == self.node_id:
            return RoutingDecision(
                next_hop=destination,
                confidence=1.0,
                alternative_hops=[],
                reasoning="Destination is self"
            )
        
        # Update network state with current metrics
        self._update_network_state(current_metrics)
        
        # Use strategy to calculate route
        decision = self._strategy.calculate_route(self.node_id, destination, self._network_state)
        
        # Update routing table with decision
        if decision.next_hop:
            self._update_routing_table_entry(destination, decision.next_hop, decision.confidence)
        
        # Publish routing event
        publish_event(EventType.ROUTE_DISCOVERED, self.node_id,
                     destination=destination, next_hop=decision.next_hop,
                     confidence=decision.confidence)
        
        return decision
    
    def update_routing_table(self, node_id: NodeID, metrics: NodeMetrics) -> None:
        """Update routing information for a node"""
        with self._lock:
            # Update node quality in network state
            quality_score = self._calculate_node_quality(metrics)
            self._network_state['node_quality'][node_id] = quality_score
            
            # Update topology (simplified - add as neighbor)
            if self.node_id not in self._network_state['topology']:
                self._network_state['topology'][self.node_id] = []
            
            if node_id not in self._network_state['topology'][self.node_id]:
                self._network_state['topology'][self.node_id].append(node_id)
        
        logger.debug(f"Updated routing info for Node {node_id}")
    
    def handle_route_failure(self, failed_route: NodeID) -> None:
        """Handle route failure notification"""
        with self._lock:
            # Remove failed route from routing table
            if failed_route in self._routing_table:
                del self._routing_table[failed_route]
            
            # Update network state
            self._network_state['problem_scores'][failed_route] = 1.0
            
            # Remove from topology
            for node_neighbors in self._network_state['topology'].values():
                if failed_route in node_neighbors:
                    node_neighbors.remove(failed_route)
        
        # Publish failure event
        publish_event(EventType.ROUTE_FAILED, self.node_id, failed_node=failed_route)
        
        logger.warning(f"Route failure handled for Node {failed_route}")
    
    def set_strategy(self, strategy: IRoutingStrategy) -> None:
        """Change routing strategy (Strategy Pattern)"""
        old_strategy = self._strategy.get_algorithm_name()
        self._strategy = strategy
        logger.info(f"Routing strategy changed from {old_strategy} to {strategy.get_algorithm_name()}")
    
    def _update_network_state(self, metrics: NodeMetrics) -> None:
        """Update network state with current metrics"""
        with self._lock:
            node_quality = self._calculate_node_quality(metrics)
            self._network_state['node_quality'][self.node_id] = node_quality
    
    def _calculate_node_quality(self, metrics: NodeMetrics) -> float:
        """Calculate node quality score from metrics"""
        # Combine different metrics into quality score
        cpu_score = max(0, 1.0 - metrics.cpu_usage / 100.0)
        memory_score = max(0, 1.0 - metrics.memory_usage / 100.0)
        latency_score = max(0, 1.0 - metrics.network_latency / 1000.0)
        
        return (cpu_score + memory_score + latency_score) / 3.0
    
    def _update_routing_table_entry(self, destination: NodeID, next_hop: NodeID, 
                                  score: float) -> None:
        """Update routing table with new route information"""
        with self._lock:
            if destination not in self._routing_table:
                self._routing_table[destination] = []
            
            routes = self._routing_table[destination]
            
            # Update existing route or add new one
            updated = False
            for i, (hop, old_score) in enumerate(routes):
                if hop == next_hop:
                    # Exponential moving average
                    new_score = 0.7 * old_score + 0.3 * score
                    routes[i] = (hop, new_score)
                    updated = True
                    break
            
            if not updated:
                routes.append((next_hop, score))
            
            # Keep only top 3 routes
            routes.sort(key=lambda x: x[1], reverse=True)
            self._routing_table[destination] = routes[:3]
    
    def get_statistics(self) -> Dict[str, any]:
        """Get routing engine statistics"""
        with self._lock:
            stats = {
                'node_id': self.node_id,
                'routing_table_size': len(self._routing_table),
                'known_nodes': len(self._network_state['node_quality']),
                'topology_links': sum(len(neighbors) for neighbors in self._network_state['topology'].values()),
                'strategy': self._strategy.get_statistics() if hasattr(self._strategy, 'get_statistics') else {}
            }
            
            return stats