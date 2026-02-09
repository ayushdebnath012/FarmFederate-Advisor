// frontend/lib/main.dart
import 'dart:convert';
import 'dart:typed_data';
import 'dart:async';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:firebase_core/firebase_core.dart';
import 'firebase_options.dart';
import 'services/auth_service.dart';
import 'constants.dart';
import 'routes.dart';
import 'theme/app_theme.dart';

// Conditional import - only import dart:html on web platform

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  try {
    // Initialize Firebase using platform-specific options
    await Firebase.initializeApp(
      options: DefaultFirebaseOptions.currentPlatform,
    );
    print('Firebase initialized successfully');
  } catch (e) {
    print('Firebase initialization error: $e');
  }
  
  runApp(const FarmFederateApp());
}

class FarmFederateApp extends StatelessWidget {
  const FarmFederateApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'FarmFederate Advisor',
      theme: AppTheme.darkTheme,
      routes: routes,
      initialRoute: '/',
      debugShowCheckedModeBanner: true,
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> with TickerProviderStateMixin {
  // Backend base URL from constants.dart
  final String apiBase = DEFAULT_BACKEND;
  final _authService = AuthService();

  late TabController _tabs;

  // Chat state
  final TextEditingController _textController = TextEditingController();
  String _adviceText = '';
  List<Map<String, dynamic>> _scores = []; // each { "label": String, "prob": double (0..1) }
  List<String> _activeLabels = [];
  String _status = 'ready';
  Uint8List? _selectedImageBytes;
  String? _selectedImageName;

  // Telemetry state (will poll /sensors/latest if available)
  Map<String, dynamic> _lastSensors = {};
  Timer? _sensorPoller;

  @override
  void initState() {
    super.initState();
    _tabs = TabController(length: 2, vsync: this);

    // start polling telemetry endpoint (if exists); safe if endpoint 404s
    _sensorPoller = Timer.periodic(const Duration(seconds: 5), (_) => _fetchLatestSensors());
  }

  @override
  void dispose() {
    _tabs.dispose();
    _textController.dispose();
    _sensorPoller?.cancel();
    super.dispose();
  }

  // -------------------------
  // Web file/camera picker
  // -------------------------
  Future<void> _pickImageWeb({bool camera = false}) async {
    if (!kIsWeb) return; // Guard against non-web platforms
    
    // Web-specific code - requires dart:html which is not available on desktop
    // To use this, run on web platform: flutter run -d chrome
    print('Web image picker requires running on web platform (flutter run -d chrome)');
    return;
  }

  // -------------------------
  // Call backend predict
  // -------------------------
  Future<void> _sendPredict() async {
    final text = _textController.text.trim();
    if (text.isEmpty && _selectedImageBytes == null) {
      setState(() {
        _status = 'Please enter text or attach an image';
      });
      return;
    }

    setState(() {
      _status = 'sending...';
      _adviceText = '';
      _scores = [];
      _activeLabels = [];
    });

    try {
      final uri = Uri.parse('$apiBase/predict');

      // If we have image bytes, send multipart/form-data; else send JSON
      http.StreamedResponse streamedResp;
      if (_selectedImageBytes != null) {
        // create multipart request
        final req = http.MultipartRequest('POST', uri);
        // text fields
        req.fields['text'] = text;
        req.fields['sensors'] = ''; // telemetry providers may add sensors; we don't send here
        req.fields['client_id'] = 'web_client';

        // image part
        final imgName = _selectedImageName ?? 'upload.jpg';
        req.files.add(http.MultipartFile.fromBytes('image', _selectedImageBytes!,
            filename: imgName)); // contentType optional

        streamedResp = await req.send().timeout(const Duration(seconds: 20));
      } else {
        // JSON POST
        final resp = await http
            .post(uri,
                headers: {'Content-Type': 'application/json'},
                body: jsonEncode({"text": text, "sensors": "", "client_id": "web_client"}))
            .timeout(const Duration(seconds: 12));
        if (resp.statusCode != 200) {
          throw Exception('HTTP ${resp.statusCode}: ${resp.body}');
        }
        final decoded = jsonDecode(resp.body) as Map<String, dynamic>;
        _handlePredictResponse(decoded);
        return;
      }

      // handle multipart streamed response
      final resp = await http.Response.fromStream(streamedResp).timeout(const Duration(seconds: 30));
      if (resp.statusCode != 200) {
        throw Exception('HTTP ${resp.statusCode}: ${resp.body}');
      }
      final decoded = jsonDecode(resp.body) as Map<String, dynamic>;
      _handlePredictResponse(decoded);
    } catch (e, st) {
      setState(() {
        _status = 'error: $e';
      });
      // print to browser console as well
      // ignore: avoid_print
      print('predict error: $e\n$st');
    }
  }

  void _handlePredictResponse(Map<String, dynamic> decoded) {
    // parse result safely
    final result = decoded['result'] as Map<String, dynamic>?;
    final advice = decoded['advice'] as String?;
    List<Map<String, dynamic>> parsedScores = [];
    List<String> parsedActive = [];

    if (result != null) {
      // parse all_scores
      final allScoresRaw = result['all_scores'];
      if (allScoresRaw is List) {
        for (final e in allScoresRaw) {
          if (e is Map) {
            final label = (e['label'] ?? e['name'] ?? e['label_name'])?.toString() ?? '';
            double prob = 0.0;
            final rawProb = e['prob'];
            if (rawProb is num) {
              prob = rawProb.toDouble();
            } else if (rawProb is String) {
              prob = double.tryParse(rawProb) ?? 0.0;
            }
            // If server returned probs in 0..100 convert to 0..1
            if (prob > 1.0) prob = prob / 100.0;
            if (label.isNotEmpty) parsedScores.add({"label": label, "prob": prob});
          }
        }
      }

      // parse active_labels: can be list of strings or list of maps
      final activeRaw = result['active_labels'];
      if (activeRaw is List) {
        for (final e in activeRaw) {
          if (e is String) {
            parsedActive.add(e);
          } else if (e is Map && e.containsKey('label')) {
            parsedActive.add(e['label'].toString());
          } else if (e is Map && e.containsKey('name')) {
            parsedActive.add(e['name'].toString());
          }
        }
      }
    }

    // If parsedScores empty, attempt to parse top-level shapes (backward compat)
    if (parsedScores.isEmpty) {
      final fallback = decoded['all_scores'] as List<dynamic>?;
      if (fallback != null) {
        for (final e in fallback) {
          if (e is Map) {
            final label = (e['label'] ?? e['name'])?.toString() ?? '';
            double prob = 0.0;
            if (e['prob'] is num) prob = (e['prob'] as num).toDouble();
            if (prob > 1.0) prob = prob / 100.0;
            if (label.isNotEmpty) parsedScores.add({"label": label, "prob": prob});
          }
        }
      }
    }

    setState(() {
      _status = 'ok';
      _adviceText = advice ?? (decoded['advice_text'] as String? ?? '');
      _scores = parsedScores;
      _activeLabels = parsedActive;
    });
  }

  // -------------------------
  // Poll device telemetry (optional)
  // -------------------------
  Future<void> _fetchLatestSensors() async {
    try {
      final resp = await http.get(Uri.parse('$apiBase/sensors/latest')).timeout(const Duration(seconds: 4));
      if (resp.statusCode == 200) {
        final decoded = jsonDecode(resp.body) as Map<String, dynamic>;
        setState(() {
          _lastSensors = decoded;
        });
      } else {
        // keep last sensors; no change
      }
    } catch (_) {
      // ignore network errors; many backends don't expose telemetry endpoint yet
    }
  }

  // -------------------------
  // UI & chips rendering
  // -------------------------
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Row(children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: Image.asset('assets/logo.png', width: 32, height: 32),
          ),
          const SizedBox(width: 12),
          const Text('FarmFederate'),
        ]),
        bottom: TabBar(controller: _tabs, tabs: const [
          Tab(text: 'Chat'),
        ]),
        actions: [
          // Qdrant Demo button
          IconButton(
            icon: const Icon(Icons.storage),
            tooltip: 'Qdrant Demo',
            onPressed: () {
              Navigator.of(context).pushNamed('/qdrant');
            },
          ),
          // User profile and logout
          PopupMenuButton<String>(
            icon: const Icon(Icons.account_circle),
            onSelected: (value) async {
              if (value == 'logout') {
                await _authService.signOut();
                if (mounted) {
                  Navigator.of(context).pushReplacementNamed('/login');
                }
              } else if (value == 'qdrant') {
                Navigator.of(context).pushNamed('/qdrant');
              }
            },
            itemBuilder: (context) => [
              PopupMenuItem(
                value: 'profile',
                child: Row(
                  children: [
                    const Icon(Icons.person, size: 20),
                    const SizedBox(width: 8),
                    Text(_authService.currentUser?.email ?? 'User'),
                  ],
                ),
              ),
              const PopupMenuDivider(),
              const PopupMenuItem(
                value: 'qdrant',
                child: Row(
                  children: [
                    Icon(Icons.storage, size: 20),
                    SizedBox(width: 8),
                    Text('Qdrant Demo'),
                  ],
                ),
              ),
              const PopupMenuItem(
                value: 'logout',
                child: Row(
                  children: [
                    Icon(Icons.logout, size: 20),
                    SizedBox(width: 8),
                    Text('Logout'),
                  ],
                ),
              ),
            ],
          ),
        ],
      ),
      body: TabBarView(controller: _tabs, children: [
        _buildChatTab(),
      ]),
    );
  }

  Widget _buildChatTab() {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(children: [
        // input
        TextField(
          controller: _textController,
          decoration: const InputDecoration(hintText: 'Describe the issue (or leave blank to send image only)'),
          minLines: 1,
          maxLines: 3,
        ),
        const SizedBox(height: 12),

        // image preview + controls
        Row(children: [
          if (_selectedImageBytes != null)
            Container(
              width: 96,
              height: 96,
              decoration: BoxDecoration(border: Border.all(color: Colors.grey.shade300)),
              child: Image.memory(_selectedImageBytes!, fit: BoxFit.cover),
            )
          else
            Container(
              width: 96,
              height: 96,
              decoration: BoxDecoration(border: Border.all(color: Colors.grey.shade300)),
              child: const Center(child: Text('No image')),
            ),
          const SizedBox(width: 12),
          Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            ElevatedButton.icon(
              onPressed: () => _onPickImage(camera: true),
              icon: const Icon(Icons.camera_alt),
              label: const Text('Camera'),
            ),
            const SizedBox(height: 8),
            ElevatedButton.icon(
              onPressed: () => _onPickImage(camera: false),
              icon: const Icon(Icons.photo_library),
              label: const Text('Upload'),
            ),
            const SizedBox(height: 8),
            TextButton(
              onPressed: () {
                setState(() {
                  _selectedImageBytes = null;
                  _selectedImageName = null;
                });
              },
              child: const Text('Clear image'),
            ),
          ])
        ]),
        const SizedBox(height: 12),

        // send
        Row(children: [
          ElevatedButton.icon(
            onPressed: _sendPredict,
            icon: const Icon(Icons.send),
            label: const Text('Send to backend'),
          ),
          const SizedBox(width: 12),
          Text('Status: $_status'),
        ]),
        const SizedBox(height: 16),
        Expanded(
            child: SingleChildScrollView(
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            if (_activeLabels.isNotEmpty) Wrap(
              spacing: 8,
              runSpacing: 6,
              children: _activeLabels.map((l) => Chip(label: Text(l))).toList(),
            ),
            const SizedBox(height: 12),
            if (_scores.isNotEmpty) ...[
              const Text('All scores:', style: TextStyle(fontWeight: FontWeight.bold)),
              const SizedBox(height: 6),
              _buildScoreChips(),
              const SizedBox(height: 12),
            ],
            const Text('Advice:', style: TextStyle(fontWeight: FontWeight.bold)),
            const SizedBox(height: 6),
            Text(_adviceText.isNotEmpty ? _adviceText : 'No advice yet'),
            const SizedBox(height: 24),
          ]),
        )),
      ]),
    );
  }

  /// Renders the list of score chips. Highlights the top-1 in red and top-2 in yellow.
  Widget _buildScoreChips() {
    if (_scores.isEmpty) return const SizedBox.shrink();

    // Defensive copy and ensure proper types
    final sorted = List<Map<String, dynamic>>.from(_scores)
      ..sort((a, b) {
        final pa = (a['prob'] is num) ? (a['prob'] as num).toDouble() : 0.0;
        final pb = (b['prob'] is num) ? (b['prob'] as num).toDouble() : 0.0;
        return pb.compareTo(pa);
      });

    // top labels
    final String? top1 = sorted.isNotEmpty ? sorted[0]['label']?.toString() : null;
    final String? top2 = (sorted.length > 1) ? sorted[1]['label']?.toString() : null;

    return Wrap(
      spacing: 12,
      runSpacing: 10,
      children: sorted.map((s) {
        final label = s['label']?.toString() ?? 'unknown';
        double prob = 0.0;
        if (s['prob'] is num) prob = (s['prob'] as num).toDouble();
        // Normalize if server used 0..100
        if (prob > 1.0) prob = prob / 100.0;
        final text = '$label: ${(prob * 100).toStringAsFixed(1)}%';

        final bool isTop1 = (label == top1);
        final bool isTop2 = (!isTop1 && label == top2);

        Color bg;
        Color fg;
        List<BoxShadow>? boxShadow;

        if (isTop1) {
          bg = Colors.red.shade600;
          fg = Colors.white;
          boxShadow = [const BoxShadow(color: Colors.black26, blurRadius: 8, offset: Offset(0, 3))];
        } else if (isTop2) {
          bg = Colors.amber.shade400;
          fg = Colors.black;
          boxShadow = [const BoxShadow(color: Colors.black12, blurRadius: 4, offset: Offset(0, 2))];
        } else {
          bg = Colors.grey.shade100;
          fg = Colors.black87;
        }

        return Container(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
          decoration: BoxDecoration(
            color: bg,
            borderRadius: BorderRadius.circular(22),
            border: Border.all(color: Colors.grey.shade300),
            boxShadow: boxShadow,
          ),
          child: Text(
            text,
            style: TextStyle(color: fg, fontWeight: (isTop1 || isTop2) ? FontWeight.bold : FontWeight.normal),
          ),
        );
      }).toList(),
    );
  }



  // Helper that dispatches to web/native pickers
  Future<void> _onPickImage({required bool camera}) async {
    if (kIsWeb) {
      await _pickImageWeb(camera: camera);
    } else {
      // Native (Android/iOS): integrate image_picker or file_picker here.
      // Example: use `image_picker` plugin, then load bytes and set _selectedImageBytes.
      // For now show a dialog telling user to add native plugin.
      showDialog(
        context: context,
        builder: (_) => AlertDialog(
          title: const Text('Native pick not implemented'),
          content: const Text('This web build handles camera/upload. For native builds add image_picker plugin and implement _pickImageNative().'),
          actions: [TextButton(onPressed: () => Navigator.of(context).pop(), child: const Text('OK'))],
        ),
      );
    }
  }
}
