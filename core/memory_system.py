# Memory system to track applied heuristics and decisions
from typing import Dict, List, Tuple, Any
import time

class MemorySystem:
    """
    Sistema de memória para armazenar e rastrear heurísticas aplicadas,
    decisões tomadas e histórico de movimentos no Sudoku.
    """
    
    def __init__(self):
        self.heuristics_used = []  # Lista de heurísticas aplicadas
        self.moves_history = []    # Histórico de movimentos
        self.decisions_log = []    # Log de decisões
        self.performance_stats = {
            'total_moves': 0,
            'successful_moves': 0,
            'heuristics_count': {},
            'start_time': None,
            'end_time': None
        }
        
    def clear_memory(self):
        """
        Limpa toda a memória do sistema
        """
        self.heuristics_used.clear()
        self.moves_history.clear()
        self.decisions_log.clear()
        self.performance_stats = {
            'total_moves': 0,
            'successful_moves': 0,
            'heuristics_count': {},
            'start_time': None,
            'end_time': None
        }
        
    def add_heuristic_usage(self, heuristic_name: str, position: Tuple[int, int], value: int):
        """
        Registra o uso de uma heurística
        
        Args:
            heuristic_name: nome da heurística usada
            position: posição (row, col) onde foi aplicada
            value: valor colocado
        """
        usage_record = {
            'heuristic': heuristic_name,
            'position': position,
            'value': value,
            'timestamp': time.time()
        }
        
        self.heuristics_used.append(usage_record)
        
        # Atualizar estatísticas
        if heuristic_name not in self.performance_stats['heuristics_count']:
            self.performance_stats['heuristics_count'][heuristic_name] = 0
        self.performance_stats['heuristics_count'][heuristic_name] += 1
        
    def add_move(self, position: Tuple[int, int], value: int, confidence: float = 1.0, success: bool = True):
        """
        Registra um movimento feito
        
        Args:
            position: posição (row, col) do movimento
            value: valor colocado
            confidence: confiança na decisão (0-1)
            success: se o movimento foi bem-sucedido
        """
        move_record = {
            'position': position,
            'value': value,
            'confidence': confidence,
            'success': success,
            'timestamp': time.time()
        }
        
        self.moves_history.append(move_record)
        self.performance_stats['total_moves'] += 1
        
        if success:
            self.performance_stats['successful_moves'] += 1
            
    def add_decision(self, decision_type: str, context: Dict[str, Any], result: Any):
        """
        Registra uma decisão tomada pelo sistema
        
        Args:
            decision_type: tipo da decisão (ex: 'heuristic_selection', 'value_choice')
            context: contexto da decisão
            result: resultado da decisão
        """
        decision_record = {
            'type': decision_type,
            'context': context,
            'result': result,
            'timestamp': time.time()
        }
        
        self.decisions_log.append(decision_record)
        
    def start_session(self):
        """
        Inicia uma nova sessão de resolução
        """
        self.performance_stats['start_time'] = time.time()
        
    def end_session(self):
        """
        Finaliza a sessão atual
        """
        self.performance_stats['end_time'] = time.time()
        
    def get_session_duration(self) -> float:
        """
        Retorna a duração da sessão em segundos
        """
        if self.performance_stats['start_time'] and self.performance_stats['end_time']:
            return self.performance_stats['end_time'] - self.performance_stats['start_time']
        return 0.0
        
    def get_heuristic_effectiveness(self, heuristic_name: str) -> Dict[str, float]:
        """
        Calcula a efetividade de uma heurística específica
        
        Args:
            heuristic_name: nome da heurística
            
        Returns:
            dicionário com estatísticas de efetividade
        """
        heuristic_moves = [h for h in self.heuristics_used if h['heuristic'] == heuristic_name]
        
        if not heuristic_moves:
            return {'usage_count': 0, 'success_rate': 0.0}
            
        # Correlacionar com movimentos bem-sucedidos
        successful_moves = 0
        for h_move in heuristic_moves:
            # Verificar se há movimento correspondente bem-sucedido
            matching_moves = [m for m in self.moves_history 
                            if m['position'] == h_move['position'] and m['success']]
            if matching_moves:
                successful_moves += 1
                
        success_rate = successful_moves / len(heuristic_moves) if heuristic_moves else 0.0
        
        return {
            'usage_count': len(heuristic_moves),
            'success_rate': success_rate,
            'avg_confidence': sum(m.get('confidence', 1.0) for m in heuristic_moves) / len(heuristic_moves)
        }
        
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Retorna um resumo completo da memória
        """
        return {
            'heuristics_count': len(self.heuristics_used),
            'moves_count': len(self.moves_history),
            'decisions_count': len(self.decisions_log),
            'performance_stats': self.performance_stats,
            'session_duration': self.get_session_duration(),
            'heuristic_breakdown': self.performance_stats['heuristics_count'],
            'success_rate': (self.performance_stats['successful_moves'] / 
                           max(self.performance_stats['total_moves'], 1)) * 100
        }
        
    def get_recent_heuristics(self, count: int = 5) -> List[Dict]:
        """
        Retorna as heurísticas mais recentemente usadas
        
        Args:
            count: número de heurísticas a retornar
            
        Returns:
            lista das heurísticas mais recentes
        """
        return self.heuristics_used[-count:] if self.heuristics_used else []
        
    def get_pattern_analysis(self) -> Dict[str, Any]:
        """
        Analisa padrões no uso de heurísticas
        """
        if not self.heuristics_used:
            return {'patterns': [], 'most_common': None}
            
        # Contar sequências de heurísticas
        sequences = []
        for i in range(len(self.heuristics_used) - 1):
            current = self.heuristics_used[i]['heuristic']
            next_h = self.heuristics_used[i + 1]['heuristic']
            sequences.append((current, next_h))
            
        # Encontrar a heurística mais comum
        heuristic_counts = self.performance_stats['heuristics_count']
        most_common = max(heuristic_counts.items(), key=lambda x: x[1]) if heuristic_counts else None
        
        return {
            'sequences': sequences,
            'most_common': most_common,
            'total_unique_heuristics': len(heuristic_counts)
        }
        
    def export_memory(self) -> Dict[str, Any]:
        """
        Exporta toda a memória para um dicionário
        """
        return {
            'heuristics_used': self.heuristics_used,
            'moves_history': self.moves_history,
            'decisions_log': self.decisions_log,
            'performance_stats': self.performance_stats,
            'summary': self.get_memory_summary(),
            'patterns': self.get_pattern_analysis()
        }
        
    def import_memory(self, memory_data: Dict[str, Any]):
        """
        Importa dados de memória de um dicionário
        
        Args:
            memory_data: dados de memória para importar
        """
        self.heuristics_used = memory_data.get('heuristics_used', [])
        self.moves_history = memory_data.get('moves_history', [])
        self.decisions_log = memory_data.get('decisions_log', [])
        self.performance_stats = memory_data.get('performance_stats', {
            'total_moves': 0,
            'successful_moves': 0,
            'heuristics_count': {},
            'start_time': None,
            'end_time': None
        }) 