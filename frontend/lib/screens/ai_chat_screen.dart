import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:file_picker/file_picker.dart';

class Message {
  final String text;
  final bool isUser;
  final DateTime timestamp;
  final Uint8List? image;
  final Map<String, dynamic>? aiAnalysis;

  Message({
    required this.text,
    required this.isUser,
    required this.timestamp,
    this.image,
    this.aiAnalysis,
  });
}

class AIChatScreen extends StatefulWidget {
  final String apiBase;

  const AIChatScreen({Key? key, required this.apiBase}) : super(key: key);

  @override
  State<AIChatScreen> createState() => _AIChatScreenState();
}

class _AIChatScreenState extends State<AIChatScreen>
    with TickerProviderStateMixin {
  final TextEditingController _controller = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  final List<Message> _messages = [];
  bool _isTyping = false;
  Uint8List? _selectedImage;
  String? _imageName;

  @override
  void initState() {
    super.initState();
    _addWelcomeMessage();
  }

  void _addWelcomeMessage() {
    setState(() {
      _messages.add(Message(
        text:
            "Hello! I'm your FarmFederate tea disease advisor. I can help diagnose leaf blight, leaf hoppers, leaf rust, looper caterpillars, and mosquito bug from symptoms and images. How can I assist you today?",
        isUser: false,
        timestamp: DateTime.now(),
      ));
    });
  }

  Future<void> _pickImage() async {
    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.image,
        withData: true,
      );

      if (result != null && result.files.isNotEmpty) {
        setState(() {
          _selectedImage = result.files.first.bytes;
          _imageName = result.files.first.name;
        });
      }
    } catch (e) {
      _showSnackBar('Error picking image: $e');
    }
  }

  Future<void> _sendMessage() async {
    if (_controller.text.trim().isEmpty && _selectedImage == null) return;

    final userText = _controller.text.trim();
    final userImage = _selectedImage;

    setState(() {
      _messages.add(Message(
        text: userText.isEmpty ? "Image uploaded" : userText,
        isUser: true,
        timestamp: DateTime.now(),
        image: userImage,
      ));
      _isTyping = true;
      _controller.clear();
      _selectedImage = null;
      _imageName = null;
    });

    _scrollToBottom();

    try {
      final response = await _callAPI(userText, userImage);

      setState(() {
        _messages.add(Message(
          text: _formatAIResponse(response),
          isUser: false,
          timestamp: DateTime.now(),
          aiAnalysis: response,
        ));
        _isTyping = false;
      });
    } catch (e) {
      setState(() {
        _messages.add(Message(
          text: "Sorry, I encountered an error: $e\n\nPlease try again.",
          isUser: false,
          timestamp: DateTime.now(),
        ));
        _isTyping = false;
      });
    }

    _scrollToBottom();
  }

  Future<Map<String, dynamic>> _callAPI(
      String text, Uint8List? imageBytes) async {
    final uri = Uri.parse('${widget.apiBase}/predict');

    if (imageBytes != null) {
      // Multipart request with image
      final request = http.MultipartRequest('POST', uri);
      request.fields['text'] = text;
      request.fields['client_id'] = 'mobile_client';
      request.files.add(http.MultipartFile.fromBytes(
        'image',
        imageBytes,
        filename: _imageName ?? 'image.jpg',
      ));

      final streamedResponse =
          await request.send().timeout(const Duration(seconds: 30));
      final response = await http.Response.fromStream(streamedResponse)
          .timeout(const Duration(seconds: 30));

      if (response.statusCode == 200) {
        return json.decode(response.body);
      } else {
        throw Exception('API error: ${response.statusCode} - ${response.body}');
      }
    } else {
      // JSON request without image
      final response = await http
          .post(
            uri,
            headers: {'Content-Type': 'application/json'},
            body: json.encode({'text': text, 'client_id': 'mobile_client'}),
          )
          .timeout(const Duration(seconds: 30));

      if (response.statusCode == 200) {
        return json.decode(response.body);
      } else {
        throw Exception('API error: ${response.statusCode} - ${response.body}');
      }
    }
  }

  String _formatAIResponse(Map<String, dynamic> response) {
    final result = response['result'] as Map<String, dynamic>?;
    final advice = response['advice'] as String? ??
        result?['advice'] as String? ??
        'Analysis complete.';

    String formatted = 'Tea Disease Analysis Results:\n\n';

    if (result != null) {
      final activeLabels = result['active_labels'] as List?;
      final allScores = (result['all_scores'] ?? result['scores']) as List?;

      if (activeLabels != null && activeLabels.isNotEmpty) {
        formatted += 'Active Tea Disease Signals:\n';
        for (var issue in activeLabels) {
          String label;
          double prob = 0.0;
          if (issue is String) {
            label = issue;
          } else if (issue is Map) {
            label = (issue['label'] ?? issue['name'] ?? 'Unknown').toString();
            final rawProb = issue['prob'];
            if (rawProb is num) prob = rawProb.toDouble();
            if (prob > 1.0) prob /= 100.0;
          } else {
            label = issue.toString();
          }
          final marker = _getIssueEmoji(label);
          final probStr =
              prob > 0 ? ': ${(prob * 100).toStringAsFixed(1)}%' : '';
          formatted += '  $marker ${_formatLabel(label)}$probStr\n';
        }
        formatted += '\n';
      } else if (allScores != null && allScores.isNotEmpty) {
        formatted += 'Confidence Ranking:\n';
        final sortedScores = List<Map<String, dynamic>>.from(allScores
            .whereType<Map>()
            .map((e) => Map<String, dynamic>.from(e)));
        sortedScores.sort((a, b) {
          final pa = (a['prob'] is num) ? (a['prob'] as num).toDouble() : 0.0;
          final pb = (b['prob'] is num) ? (b['prob'] as num).toDouble() : 0.0;
          return pb.compareTo(pa);
        });

        for (int i = 0; i < sortedScores.length && i < 3; i++) {
          final score = sortedScores[i];
          final label =
              (score['label'] ?? score['name'] ?? 'Unknown').toString();
          double prob =
              (score['prob'] is num) ? (score['prob'] as num).toDouble() : 0.0;
          if (prob > 1.0) prob /= 100.0;
          final marker = _getIssueEmoji(label);
          formatted +=
              '  $marker ${_formatLabel(label)}: ${(prob * 100).toStringAsFixed(1)}%\n';
        }
        formatted += '\n';
      }
    }

    formatted += 'Recommendations:\n$advice';

    return formatted;
  }

  String _getIssueEmoji(String label) {
    final normalized = label.toUpperCase();
    if (normalized.contains('LEAF_BLIGHT')) return '[BLIGHT]';
    if (normalized.contains('LEAF_HOPPERS')) return '[HOPPER]';
    if (normalized.contains('LEAF_RUST')) return '[RUST]';
    if (normalized.contains('LOOPER')) return '[LOOPER]';
    if (normalized.contains('MOSQUITO')) return '[MOSQUITO]';
    return '[TEA]';
  }

  String _formatLabel(String label) {
    return label
        .split('_')
        .map((word) => word[0].toUpperCase() + word.substring(1).toLowerCase())
        .join(' ');
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  void _showSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), duration: const Duration(seconds: 2)),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0A0E21),
      appBar: AppBar(
        elevation: 0,
        backgroundColor: const Color(0xFF1D1E33),
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                  colors: [Color(0xFF667EEA), Color(0xFF764BA2)],
                ),
                borderRadius: BorderRadius.circular(8),
              ),
              child: const Icon(Icons.psychology, size: 24),
            ),
            const SizedBox(width: 12),
            const Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'AI Farm Advisor',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF93F9B9),
                  ),
                ),
                Text(
                  'Powered by Multimodal AI',
                  style: TextStyle(
                    fontSize: 12,
                    color: Color(0xFF00BCD4),
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ],
            ),
          ],
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.delete_outline),
            onPressed: () {
              setState(() {
                _messages.clear();
                _addWelcomeMessage();
              });
            },
          ),
        ],
      ),
      body: Column(
        children: [
          if (_messages.isEmpty) _buildEmptyState() else _buildMessageList(),
          if (_isTyping) _buildTypingIndicator(),
          _buildInputArea(),
        ],
      ),
    );
  }

  Widget _buildEmptyState() {
    return Expanded(
      child: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              padding: const EdgeInsets.all(24),
              decoration: const BoxDecoration(
                gradient: LinearGradient(
                  colors: [Color(0xFF667EEA), Color(0xFF764BA2)],
                ),
                shape: BoxShape.circle,
              ),
              child:
                  const Icon(Icons.agriculture, size: 64, color: Colors.white),
            ),
            const SizedBox(height: 24),
            const Text(
              'AI Farm Advisor',
              style: TextStyle(
                color: Colors.white,
                fontSize: 28,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 12),
            Text(
              'Ask me anything about your crops',
              style: TextStyle(
                color: Colors.white.withValues(alpha: 0.6),
                fontSize: 16,
              ),
            ),
            const SizedBox(height: 32),
            _buildQuickSuggestions(),
          ],
        ),
      ),
    );
  }

  Widget _buildQuickSuggestions() {
    final suggestions = [
      'Leaf blight treatment',
      'Leaf rust symptoms',
      'Looper caterpillar damage',
      'Mosquito bug control',
    ];

    return Wrap(
      spacing: 8,
      runSpacing: 8,
      alignment: WrapAlignment.center,
      children: suggestions.map((text) => _buildSuggestionChip(text)).toList(),
    );
  }

  Widget _buildSuggestionChip(String text) {
    return InkWell(
      onTap: () {
        _controller.text = text;
      },
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        decoration: BoxDecoration(
          color: const Color(0xFF1D1E33),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: Colors.white.withValues(alpha: 0.2)),
        ),
        child: Text(
          text,
          style: const TextStyle(color: Colors.white70, fontSize: 13),
        ),
      ),
    );
  }

  Widget _buildMessageList() {
    return Expanded(
      child: ListView.builder(
        controller: _scrollController,
        padding: const EdgeInsets.all(16),
        itemCount: _messages.length,
        itemBuilder: (context, index) => _buildMessageBubble(_messages[index]),
      ),
    );
  }

  Widget _buildMessageBubble(Message message) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 16),
      child: Row(
        mainAxisAlignment:
            message.isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (!message.isUser) _buildAvatar(false),
          const SizedBox(width: 8),
          Flexible(
            child: Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                gradient: message.isUser
                    ? const LinearGradient(
                        colors: [Color(0xFF667EEA), Color(0xFF764BA2)],
                      )
                    : null,
                color: message.isUser ? null : Colors.white,
                borderRadius: BorderRadius.circular(16),
                boxShadow: [
                  BoxShadow(
                    color:
                        (message.isUser ? const Color(0xFF667EEA) : Colors.grey)
                            .withValues(alpha: 0.3),
                    blurRadius: 8,
                    offset: const Offset(0, 4),
                  ),
                ],
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  if (message.image != null) ...[
                    ClipRRect(
                      borderRadius: BorderRadius.circular(8),
                      child: Image.memory(
                        message.image!,
                        height: 200,
                        fit: BoxFit.cover,
                      ),
                    ),
                    const SizedBox(height: 12),
                  ],
                  _buildFormattedText(message.text, message.isUser),
                  if (message.aiAnalysis != null) ...[
                    const SizedBox(height: 8),
                    _buildAnalysisChips(message.aiAnalysis!),
                  ],
                  const SizedBox(height: 8),
                  Text(
                    _formatTime(message.timestamp),
                    style: TextStyle(
                      color: message.isUser
                          ? Colors.white.withValues(alpha: 0.5)
                          : Colors.black45,
                      fontSize: 11,
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(width: 8),
          if (message.isUser) _buildAvatar(true),
        ],
      ),
    );
  }

  Widget _buildAvatar(bool isUser) {
    return Container(
      width: 36,
      height: 36,
      decoration: BoxDecoration(
        gradient: isUser
            ? const LinearGradient(
                colors: [Color(0xFF667EEA), Color(0xFF764BA2)])
            : const LinearGradient(
                colors: [Color(0xFF1D976C), Color(0xFF93F9B9)]),
        shape: BoxShape.circle,
      ),
      child: Icon(
        isUser ? Icons.person : Icons.smart_toy,
        color: Colors.white,
        size: 20,
      ),
    );
  }

  Widget _buildAnalysisChips(Map<String, dynamic> analysis) {
    final result = analysis['result'] as Map<String, dynamic>?;
    if (result == null) return const SizedBox.shrink();

    // Backend uses 'all_scores' key, fallback to 'scores'
    final scores = (result['all_scores'] ?? result['scores']) as List?;
    if (scores == null || scores.isEmpty) return const SizedBox.shrink();

    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: scores.take(3).map((score) {
        final label = score['label'] ?? '';
        final prob = ((score['prob'] ?? 0.0) * 100).toStringAsFixed(0);
        return Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
          decoration: BoxDecoration(
            color: Colors.red.shade50,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: Colors.red.shade300, width: 1.5),
          ),
          child: Text(
            '${_formatLabel(label)}: $prob%',
            style: TextStyle(
                color: Colors.red.shade700,
                fontSize: 12,
                fontWeight: FontWeight.bold),
          ),
        );
      }).toList(),
    );
  }

  Widget _buildFormattedText(String text, bool isUser) {
    if (isUser) {
      return Text(
        text,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 15,
          height: 1.4,
        ),
      );
    }

    // Parse and format AI responses with colored titles
    final lines = text.split('\n');
    final widgets = <Widget>[];

    // Extract and sort issues by probability
    final issueLines = <MapEntry<String, double>>[];
    final nonIssueLines = <String>[];
    bool inIssuesSection = false;

    for (final line in lines) {
      if (line.contains('Active Tea Disease Signals:')) {
        inIssuesSection = true;
        nonIssueLines.add(line);
        continue;
      } else if (line.contains('Recommendations:') ||
          line.contains('Confidence Ranking:')) {
        inIssuesSection = false;
        nonIssueLines.add(line);
        continue;
      }

      if (inIssuesSection && line.trim().startsWith('[')) {
        // Extract percentage from line (e.g., "67.6%" -> 67.6)
        final percentMatch = RegExp(r'(\d+\.?\d*)%').firstMatch(line);
        final probability =
            percentMatch != null ? double.parse(percentMatch.group(1)!) : 0.0;
        issueLines.add(MapEntry(line, probability));
      } else {
        nonIssueLines.add(line);
      }
    }

    // Sort issues by probability (highest first)
    issueLines.sort((a, b) => b.value.compareTo(a.value));

    // Build widgets with sorted issues
    int globalIssueCount = 0;

    for (final line in nonIssueLines) {
      if (line.trim().isEmpty) {
        widgets.add(const SizedBox(height: 8));
        continue;
      }

      // Check if this is where issues should be inserted
      if (line.contains('Active Tea Disease Signals:')) {
        // Add the header
        widgets.add(
          Padding(
            padding: const EdgeInsets.only(bottom: 6),
            child: Text(
              line,
              style: const TextStyle(
                color: Color(0xFF00BCD4),
                fontSize: 17,
                fontWeight: FontWeight.bold,
                height: 1.6,
              ),
            ),
          ),
        );

        // Add sorted issues
        for (final issue in issueLines) {
          TextStyle issueStyle;
          if (globalIssueCount == 0) {
            // Highest probability - RED
            issueStyle = const TextStyle(
              color: Colors.red,
              fontSize: 17,
              fontWeight: FontWeight.bold,
              height: 1.5,
            );
          } else if (globalIssueCount == 1) {
            // Second highest - YELLOW
            issueStyle = const TextStyle(
              color: Colors.amber,
              fontSize: 17,
              fontWeight: FontWeight.bold,
              height: 1.5,
            );
          } else {
            // Others - BLACK
            issueStyle = const TextStyle(
              color: Colors.black87,
              fontSize: 15,
              fontWeight: FontWeight.w500,
              height: 1.5,
            );
          }

          widgets.add(
            Padding(
              padding: const EdgeInsets.only(bottom: 6),
              child: Text(issue.key, style: issueStyle),
            ),
          );
          globalIssueCount++;
        }
        continue;
      }

      TextStyle style;
      if (line.contains('Tea Disease Analysis Results:') ||
          line.contains('Confidence Ranking:') ||
          line.contains('Recommendations:')) {
        // Section headers in cyan
        style = const TextStyle(
          color: Color(0xFF00BCD4),
          fontSize: 17,
          fontWeight: FontWeight.bold,
          height: 1.6,
        );
      } else if (line.trim().startsWith('-')) {
        // Recommendation items in darker text
        style = const TextStyle(
          color: Colors.black87,
          fontSize: 15,
          height: 1.5,
        );
      } else {
        // Normal text in black
        style = const TextStyle(
          color: Colors.black87,
          fontSize: 15,
          height: 1.4,
        );
      }

      widgets.add(
        Padding(
          padding: const EdgeInsets.only(bottom: 6),
          child: Text(line, style: style),
        ),
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: widgets,
    );
  }

  Widget _buildTypingIndicator() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      child: Row(
        children: [
          _buildAvatar(false),
          const SizedBox(width: 12),
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(16),
              boxShadow: [
                BoxShadow(
                  color: Colors.grey.withValues(alpha: 0.3),
                  blurRadius: 8,
                  offset: const Offset(0, 4),
                ),
              ],
            ),
            child: Row(
              children: [
                _buildDot(0),
                const SizedBox(width: 4),
                _buildDot(1),
                const SizedBox(width: 4),
                _buildDot(2),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDot(int index) {
    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0.0, end: 1.0),
      duration: const Duration(milliseconds: 600),
      builder: (context, value, child) {
        final delay = index * 0.2;
        final animValue = ((value - delay) * 2).clamp(0.0, 1.0);
        return Container(
          width: 8,
          height: 8,
          decoration: BoxDecoration(
            color: Colors.grey.withValues(alpha: 0.3 + (animValue * 0.5)),
            shape: BoxShape.circle,
          ),
        );
      },
      onEnd: () => setState(() {}),
    );
  }

  Widget _buildInputArea() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1D1E33),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.3),
            blurRadius: 10,
            offset: const Offset(0, -5),
          ),
        ],
      ),
      child: Column(
        children: [
          if (_selectedImage != null) _buildImagePreview(),
          Row(
            children: [
              IconButton(
                icon: const Icon(Icons.image, color: Color(0xFF667EEA)),
                onPressed: _pickImage,
              ),
              Expanded(
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  decoration: BoxDecoration(
                    color: const Color(0xFF0A0E21),
                    borderRadius: BorderRadius.circular(24),
                    border: Border.all(
                        color: const Color(0xFF1D976C).withValues(alpha: 0.5)),
                  ),
                  child: TextField(
                    controller: _controller,
                    style: const TextStyle(color: Colors.white, fontSize: 15),
                    decoration: InputDecoration(
                      hintText: 'Ask about your crops...',
                      hintStyle:
                          TextStyle(color: Colors.white.withValues(alpha: 0.5)),
                      border: InputBorder.none,
                      filled: false,
                    ),
                    textInputAction: TextInputAction.send,
                    onSubmitted: (_) => _sendMessage(),
                  ),
                ),
              ),
              const SizedBox(width: 8),
              Container(
                decoration: const BoxDecoration(
                  gradient: LinearGradient(
                    colors: [Color(0xFF667EEA), Color(0xFF764BA2)],
                  ),
                  shape: BoxShape.circle,
                ),
                child: IconButton(
                  icon: const Icon(Icons.send, color: Colors.white),
                  onPressed: _sendMessage,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildImagePreview() {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: const Color(0xFF0A0E21),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: Image.memory(
              _selectedImage!,
              width: 60,
              height: 60,
              fit: BoxFit.cover,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              _imageName ?? 'Image selected',
              style: const TextStyle(color: Colors.white70, fontSize: 13),
              overflow: TextOverflow.ellipsis,
            ),
          ),
          IconButton(
            icon: const Icon(Icons.close, color: Colors.white54),
            onPressed: () {
              setState(() {
                _selectedImage = null;
                _imageName = null;
              });
            },
          ),
        ],
      ),
    );
  }

  String _formatTime(DateTime time) {
    final now = DateTime.now();
    final diff = now.difference(time);

    if (diff.inMinutes < 1) return 'Just now';
    if (diff.inHours < 1) return '${diff.inMinutes}m ago';
    if (diff.inDays < 1) return '${diff.inHours}h ago';
    return '${time.hour}:${time.minute.toString().padLeft(2, '0')}';
  }

  @override
  void dispose() {
    _controller.dispose();
    _scrollController.dispose();
    super.dispose();
  }
}
