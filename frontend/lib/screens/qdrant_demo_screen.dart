import 'package:flutter/material.dart';
import '../services/api_service.dart';

/// QdrantDemoScreen - Demonstrates Qdrant Search, Memory, and Recommendations
/// for the Convolve 4.0 Hackathon demo video
class QdrantDemoScreen extends StatefulWidget {
  final String apiBase;
  const QdrantDemoScreen({super.key, required this.apiBase});

  @override
  State<QdrantDemoScreen> createState() => _QdrantDemoScreenState();
}

class _QdrantDemoScreenState extends State<QdrantDemoScreen> {
  late ApiService _api;
  bool _isLoading = false;
  String _statusMessage = "Ready to demo Qdrant integration";

  // Demo populate results
  List<dynamic> _populatedIds = [];

  // Search results
  List<dynamic> _searchResults = [];

  // Selected vector type for search
  String _vectorType = 'visual';

  @override
  void initState() {
    super.initState();
    _api = ApiService(widget.apiBase);
    _checkHealth();
  }

  Future<void> _checkHealth() async {
    setState(() => _isLoading = true);
    try {
      // Simple health check - just verify we can reach the backend
      await _api.demoPopulate(n: 0).catchError((_) => <String, dynamic>{});
      setState(() {
        _statusMessage = "Backend connected at ${widget.apiBase}";
      });
    } catch (e) {
      setState(() => _statusMessage = "Backend error: $e");
    } finally {
      setState(() => _isLoading = false);
    }
  }

  Future<void> _demoPopulate() async {
    setState(() {
      _isLoading = true;
      _statusMessage = "Populating Qdrant with demo vectors...";
    });
    try {
      final result = await _api.demoPopulate(n: 5);
      setState(() {
        _populatedIds = result['ids'] ?? [];
        _statusMessage =
            "Populated ${_populatedIds.length} vectors into Qdrant";
      });
    } catch (e) {
      setState(() => _statusMessage = "Populate error: $e");
    } finally {
      setState(() => _isLoading = false);
    }
  }

  Future<void> _demoSearch() async {
    setState(() {
      _isLoading = true;
      _statusMessage = "Searching Qdrant with $_vectorType vectors...";
    });
    try {
      final result = await _api.demoSearch(topK: 5, vectorType: _vectorType);
      setState(() {
        _searchResults = result['hits'] ?? [];
        _statusMessage = "Found ${_searchResults.length} similar cases";
      });
    } catch (e) {
      setState(() => _statusMessage = "Search error: $e");
    } finally {
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Qdrant Demo - FarmFederate'),
        backgroundColor: Colors.green.shade700,
        foregroundColor: Colors.white,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header
            Card(
              color: Colors.green.shade50,
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(Icons.storage,
                            color: Colors.green.shade700, size: 32),
                        const SizedBox(width: 12),
                        const Expanded(
                          child: Text(
                            'Qdrant Vector Database Demo',
                            style: TextStyle(
                                fontSize: 20, fontWeight: FontWeight.bold),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    const Text(
                      'Demonstrating Search, Memory, and Recommendations for Convolve 4.0',
                      style: TextStyle(fontSize: 14, color: Colors.grey),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 16),

            // Status
            Card(
              child: ListTile(
                leading: _isLoading
                    ? const SizedBox(
                        width: 24,
                        height: 24,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : Icon(
                        _statusMessage.contains('error')
                            ? Icons.error
                            : Icons.check_circle,
                        color: _statusMessage.contains('error')
                            ? Colors.red
                            : Colors.green,
                      ),
                title: Text(_statusMessage),
                subtitle: Text('API: ${widget.apiBase}'),
              ),
            ),

            const SizedBox(height: 24),

            // Demo Actions Section
            const Text(
              '1. SEARCH: Populate Knowledge Base',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            const Text(
              'Creates demo vectors in crop_health_knowledge collection with visual (512-d CLIP) and semantic (384-d SBERT) embeddings.',
              style: TextStyle(fontSize: 12, color: Colors.grey),
            ),
            const SizedBox(height: 8),
            ElevatedButton.icon(
              onPressed: _isLoading ? null : _demoPopulate,
              icon: const Icon(Icons.add_circle),
              label: const Text('Populate Qdrant Demo Data'),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.blue,
                foregroundColor: Colors.white,
              ),
            ),

            if (_populatedIds.isNotEmpty) ...[
              const SizedBox(height: 8),
              Card(
                color: Colors.blue.shade50,
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Populated IDs: ${_populatedIds.length}',
                          style: const TextStyle(fontWeight: FontWeight.bold)),
                      const SizedBox(height: 4),
                      Text(
                          _populatedIds
                              .map((id) => id.toString().substring(0, 8))
                              .join(', '),
                          style: const TextStyle(
                              fontSize: 11, fontFamily: 'monospace')),
                    ],
                  ),
                ),
              ),
            ],

            const SizedBox(height: 24),

            // Vector Search Section
            const Text(
              '2. SEARCH: Vector Similarity Search',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            const Text(
              'Searches Qdrant using the last populated vector. Uses named vectors for multimodal search.',
              style: TextStyle(fontSize: 12, color: Colors.grey),
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                ChoiceChip(
                  label: const Text('Visual (CLIP)'),
                  selected: _vectorType == 'visual',
                  onSelected: (selected) {
                    if (selected) setState(() => _vectorType = 'visual');
                  },
                ),
                const SizedBox(width: 8),
                ChoiceChip(
                  label: const Text('Semantic (SBERT)'),
                  selected: _vectorType == 'semantic',
                  onSelected: (selected) {
                    if (selected) setState(() => _vectorType = 'semantic');
                  },
                ),
              ],
            ),
            const SizedBox(height: 8),
            ElevatedButton.icon(
              onPressed: _isLoading ? null : _demoSearch,
              icon: const Icon(Icons.search),
              label: Text('Search with $_vectorType vectors'),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.green,
                foregroundColor: Colors.white,
              ),
            ),

            if (_searchResults.isNotEmpty) ...[
              const SizedBox(height: 16),
              const Text(
                '3. RECOMMENDATIONS: Retrieved Cases',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 8),
              ..._searchResults
                  .map((hit) => _buildSearchResultCard(hit))
                  .toList(),
            ],

            const SizedBox(height: 24),

            // Memory Section
            Card(
              color: Colors.orange.shade50,
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(Icons.memory, color: Colors.orange.shade700),
                        const SizedBox(width: 8),
                        const Text('MEMORY: Session History',
                            style: TextStyle(
                                fontSize: 16, fontWeight: FontWeight.bold)),
                      ],
                    ),
                    const SizedBox(height: 8),
                    const Text(
                      'The farm_session_memory collection stores diagnosis history per farm_id, enabling personalized recommendations over time.',
                      style: TextStyle(fontSize: 12),
                    ),
                    const SizedBox(height: 8),
                    const Text(
                      'Collections:\n'
                      '• crop_health_knowledge: 512-d visual + 384-d semantic\n'
                      '• farm_session_memory: 384-d semantic with farm_id filter',
                      style: TextStyle(fontSize: 11, fontFamily: 'monospace'),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 24),

            // Architecture Info
            Card(
              color: Colors.purple.shade50,
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(Icons.architecture, color: Colors.purple.shade700),
                        const SizedBox(width: 8),
                        const Text('Architecture',
                            style: TextStyle(
                                fontSize: 16, fontWeight: FontWeight.bold)),
                      ],
                    ),
                    const SizedBox(height: 8),
                    const Text(
                      'Flutter App → FastAPI → Qdrant\n\n'
                      'Named Vectors:\n'
                      '• visual: CLIP ViT-B/32 (512 dimensions)\n'
                      '• semantic: all-MiniLM-L6-v2 (384 dimensions)\n\n'
                      'Distance Metric: COSINE\n'
                      'Payload: stress_type, crop_name, severity, source',
                      style: TextStyle(fontSize: 12, fontFamily: 'monospace'),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSearchResultCard(Map<String, dynamic> hit) {
    final payload = hit['payload'] as Map<String, dynamic>? ?? {};
    final score = (hit['score'] as num?)?.toStringAsFixed(4) ?? 'N/A';

    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      child: ListTile(
        leading: CircleAvatar(
          backgroundColor: _getStressColor(payload['stress_type']),
          child: Text(
            (payload['severity']?.toString() ?? '?'),
            style: const TextStyle(
                color: Colors.white, fontWeight: FontWeight.bold),
          ),
        ),
        title: Text(payload['stress_type'] ?? 'Unknown Disease'),
        subtitle: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Crop: ${payload['crop_name'] ?? 'unknown'}'),
            Text('Source: ${payload['source'] ?? 'unknown'}'),
            Text('Score: $score',
                style: const TextStyle(fontWeight: FontWeight.bold)),
          ],
        ),
        isThreeLine: true,
      ),
    );
  }

  Color _getStressColor(String? stressType) {
    switch (stressType) {
      case 'LEAF_BLIGHT':
        return Colors.green;
      case 'LEAF_HOPPERS':
        return Colors.red;
      case 'LEAF_RUST':
        return Colors.blue;
      case 'LOOPER_CATERPILLARS':
        return Colors.purple;
      case 'MOSQUITO_BUG':
        return Colors.teal;
      default:
        return Colors.grey;
    }
  }
}
