// frontend/lib/screens/chat_screen.dart
// ChatScreen (text + optional image) + ApiService
// - Handles mobile + web image picking (ImagePicker for camera/mobile, FilePicker for upload).
// - Highlights top-2 scores: top1 -> red, top2 -> yellow.
// - Sends multipart (file bytes) if image present, otherwise JSON POST.

import 'dart:convert';
import 'dart:typed_data';
import 'dart:io' show File; // only used on non-web
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:file_picker/file_picker.dart';
import 'package:farmfederate_advisor/services/api_service.dart';

class ChatScreen extends StatefulWidget {
  final String apiBase; // e.g. "http://10.0.2.2:8000"
  const ChatScreen({Key? key, required this.apiBase}) : super(key: key);

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final TextEditingController _ctrl = TextEditingController();

  // For mobile: store a File
  File? _imageFile;

  // For web (and unified display) we store bytes + filename if chosen via file picker
  Uint8List? _imageBytes;
  String? _imageName;

  bool _loading = false;
  String _status = "idle";
  List<Map<String, dynamic>> _scores =
      []; // each: {label: String, prob: double}
  String _advice = "";
  Map<String, dynamic> _debug = {};

  final ImagePicker _picker = ImagePicker();

  void _setStatus(String s) => setState(() => _status = s);

  // Use camera (mobile/desktop where supported). ImagePicker on web will not capture file, so guard.
  Future<void> _pickCamera() async {
    try {
      if (kIsWeb) {
        // camera via web is not supported by image_picker package in many setups.
        _setStatus("camera not supported on web - use Upload");
        return;
      }
      final XFile? photo =
          await _picker.pickImage(source: ImageSource.camera, imageQuality: 80);
      if (photo == null) return;
      setState(() {
        _imageFile = File(photo.path);
        _imageBytes = null;
        _imageName = photo.name;
      });
    } catch (e) {
      _setStatus("camera error");
      debugPrint("camera error: $e");
    }
  }

  // File picker (works on web and mobile)
  Future<void> _pickFile() async {
    try {
      final res = await FilePicker.platform
          .pickFiles(type: FileType.image, withData: true);
      if (res == null || res.files.isEmpty) return;
      final picked = res.files.first;
      if (kIsWeb) {
        // On web we have bytes directly
        setState(() {
          _imageBytes = picked.bytes;
          _imageName = picked.name;
          _imageFile = null;
        });
      } else {
        // On mobile we may get a path as well
        final p = picked.path;
        if (p != null) {
          setState(() {
            _imageFile = File(p);
            _imageBytes = null;
            _imageName = picked.name;
          });
        } else if (picked.bytes != null) {
          // fallback: write bytes to temp file...
          setState(() {
            _imageBytes = picked.bytes;
            _imageFile = null;
            _imageName = picked.name;
          });
        }
      }
    } catch (e) {
      _setStatus("file pick error");
      debugPrint("file pick error: $e");
    }
  }

  void _clearImage() {
    setState(() {
      _imageFile = null;
      _imageBytes = null;
      _imageName = null;
    });
  }

  Future<void> _send() async {
    final text = _ctrl.text.trim();
    if (text.isEmpty && _imageFile == null && _imageBytes == null) {
      _setStatus("provide text or image");
      return;
    }

    setState(() {
      _loading = true;
      _setStatus("sending...");
      _scores = [];
      _advice = "";
      _debug = {};
    });

    try {
      final api = ApiService(widget.apiBase);
      final resp = await api.predict(
        text: text,
        imagePath: _imageFile?.path,
        imageBytes: _imageBytes,
        imageName: _imageName,
        clientId: "flutter_client",
      );

      final result = resp['result'] ?? resp['data'] ?? resp;
      final allScores = result['all_scores'] as List<dynamic>? ??
          result['scores'] as List<dynamic>?;

      final parsedScores = (allScores ?? []).map<Map<String, dynamic>>((e) {
        final label = (e['label'] ?? e['name'] ?? "unknown").toString();
        double prob = 0.0;
        if (e['prob'] is num) {
          prob = (e['prob'] as num).toDouble();
        } else {
          prob = double.tryParse("${e['prob']}") ?? 0.0;
        }
        // convert 0..100 -> 0..1 if needed
        if (prob > 1.0) prob = prob / 100.0;
        return {"label": label, "prob": prob.clamp(0.0, 1.0)};
      }).toList();

      setState(() {
        _scores = parsedScores;
        // Defensive advice parsing
        _advice =
            (resp['advice'] ?? result['advice'] ?? resp['advice_text'] ?? "")
                    ?.toString() ??
                "";
        _debug = resp['debug'] ?? {};
        _setStatus("ok");
      });
    } catch (e, st) {
      _setStatus("error");
      debugPrint("predict error: $e\n$st");
      setState(() {
        _advice = "Error: $e";
      });
    } finally {
      setState(() {
        _loading = false;
      });
    }
  }

  // RAG diagnose handler
  Future<void> _sendRag() async {
    final description = _ctrl.text.trim();
    if (description.isEmpty && _imageFile == null && _imageBytes == null) {
      _setStatus("provide text or image");
      return;
    }

    setState(() {
      _loading = true;
      _setStatus("running rag...");
      _scores = [];
      _advice = "";
      _debug = {};
    });

    try {
      final api = ApiService(widget.apiBase);
      final resp = await api.ragDiagnose(
        description: description,
        imagePath: _imageFile?.path,
        imageBytes: _imageBytes,
        imageName: _imageName,
        clientId: "flutter_client",
      );

      final res = resp['result'] ?? resp;
      final retrieved = res['retrieved'] as List<dynamic>? ?? [];
      final prompt = res['prompt']?.toString() ?? '';
      final treatment = res['treatment'];

      setState(() {
        // If treatment is a string (mock LLM), show as advice; if dict, show prompt in debug
        if (treatment is String) {
          _advice = treatment;
        } else if (treatment is Map) {
          _debug['llm'] = treatment;
          _advice = treatment['text']?.toString() ?? prompt;
        } else {
          _debug['prompt'] = prompt;
          _advice = prompt;
        }

        _debug['retrieved'] = retrieved;
        _setStatus("ok");
      });
    } catch (e, st) {
      _setStatus("error");
      debugPrint("rag error: $e\n$st");
      setState(() {
        _advice = "Error: $e";
      });
    } finally {
      setState(() {
        _loading = false;
      });
    }
  }

  // ---------------- Demo helpers ----------------
  Future<void> _demoPopulate() async {
    setState(() {
      _loading = true;
      _setStatus("populating demo...");
      _debug = {};
      _advice = "";
    });
    try {
      final api = ApiService(widget.apiBase);
      final resp = await api.demoPopulate(n: 5);
      setState(() {
        _debug['demo_populate'] = resp;
        _advice =
            "Populated collection: ${resp['collection']} with ${resp['ids']?.length ?? 0} items";
        _setStatus('ok');
      });
    } catch (e) {
      _setStatus('error');
      setState(() {
        _advice = 'Populate failed: $e';
      });
    } finally {
      setState(() {
        _loading = false;
      });
    }
  }

  Future<void> _demoSearch() async {
    setState(() {
      _loading = true;
      _setStatus("searching demo...");
      _debug = {};
      _advice = "";
    });
    try {
      final api = ApiService(widget.apiBase);
      final resp = await api.demoSearch(topK: 3, vectorType: 'visual');
      setState(() {
        _debug['demo_search'] = resp;
        final hits = resp['hits'] as List<dynamic>? ?? [];
        _advice = 'Found ${hits.length} hits (see Debug)';
        _setStatus('ok');
      });
    } catch (e) {
      _setStatus('error');
      setState(() {
        _advice = 'Search failed: $e';
      });
    } finally {
      setState(() {
        _loading = false;
      });
    }
  }

  // Build chips; top1 red, top2 yellow.
  Widget _buildScoreChips() {
    if (_scores.isEmpty) return const SizedBox.shrink();

    final sorted = List<Map<String, dynamic>>.from(_scores)
      ..sort((a, b) => (b['prob'] as num).compareTo(a['prob'] as num));

    debugPrint('SCORES (sorted): $sorted');

    final String? top1 =
        sorted.isNotEmpty ? sorted[0]['label'] as String : null;
    final String? top2 =
        sorted.length > 1 ? sorted[1]['label'] as String : null;

    return Wrap(
      spacing: 12,
      runSpacing: 12,
      children: sorted.map((s) {
        final label = s['label'] as String;
        final prob = (s['prob'] as num).toDouble();
        final scoreText = "$label: ${(prob * 100).toStringAsFixed(1)}%";

        final isTop1 = label == top1;
        final isTop2 = label == top2;

        Color bgColor;
        Color textColor;
        double elevation = 0;

        if (isTop1) {
          bgColor = Colors.red.shade700;
          textColor = Colors.white;
          elevation = 6;
        } else if (isTop2) {
          bgColor = Colors.amber.shade700;
          textColor = Colors.black;
          elevation = 4;
        } else {
          bgColor = Colors.grey.shade100;
          textColor = Colors.black87;
          elevation = 0;
        }

        // Material ensures color & elevation appear correctly on web/desktop
        return Material(
          color: bgColor,
          elevation: elevation,
          borderRadius: BorderRadius.circular(22),
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
            child: Text(
              scoreText,
              style: TextStyle(
                color: textColor,
                fontWeight:
                    (isTop1 || isTop2) ? FontWeight.bold : FontWeight.normal,
              ),
            ),
          ),
        );
      }).toList(),
    );
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    Widget imagePreview;
    if (_imageBytes != null) {
      imagePreview = Image.memory(_imageBytes!,
          width: 120, height: 120, fit: BoxFit.cover);
    } else if (_imageFile != null) {
      imagePreview =
          Image.file(_imageFile!, width: 120, height: 120, fit: BoxFit.cover);
    } else {
      imagePreview = Container(
        width: 120,
        height: 120,
        color: Colors.grey.shade100,
        child: const Center(child: Text("No image")),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Row(children: [
          Icon(Icons.agriculture, size: 28),
          SizedBox(width: 8),
          Text("FarmFederate"),
        ]),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(18),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          TextField(
            controller: _ctrl,
            minLines: 1,
            maxLines: 4,
            decoration: const InputDecoration(
              hintText: "Describe the issue (or leave blank for image-only)",
              border: UnderlineInputBorder(),
            ),
          ),
          const SizedBox(height: 18),

          // image preview + buttons
          Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
            ClipRRect(
                borderRadius: BorderRadius.circular(6), child: imagePreview),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    ElevatedButton.icon(
                      onPressed: _pickCamera,
                      icon: const Icon(Icons.camera_alt),
                      label: const Text("Camera"),
                    ),
                    const SizedBox(height: 8),
                    ElevatedButton.icon(
                      onPressed: _pickFile,
                      icon: const Icon(Icons.upload_file),
                      label: const Text("Upload"),
                    ),
                    TextButton(
                        onPressed: _clearImage,
                        child: const Text("Clear image")),
                  ]),
            ),
          ]),
          const SizedBox(height: 18),

          Row(children: [
            ElevatedButton.icon(
              onPressed: _loading ? null : _send,
              icon: const Icon(Icons.send),
              label: Text(_loading ? "Sending..." : "Send to backend"),
            ),
            const SizedBox(width: 12),
            ElevatedButton.icon(
              onPressed: _loading ? null : _sendRag,
              icon: const Icon(Icons.grain),
              label: Text(_loading ? "Running RAG..." : "RAG Diagnose"),
            ),
            const SizedBox(width: 12),
            ElevatedButton.icon(
              onPressed: _loading ? null : _demoPopulate,
              icon: const Icon(Icons.cloud_upload),
              label: Text(_loading ? "Populating..." : "Populate Demo"),
            ),
            const SizedBox(width: 12),
            ElevatedButton.icon(
              onPressed: _loading ? null : _demoSearch,
              icon: const Icon(Icons.search),
              label: Text(_loading ? "Searching..." : "Search Demo"),
            ),
            const SizedBox(width: 12),
            Text("Status: $_status"),
          ]),

          const SizedBox(height: 18),

          // TEMP debug button to verify coloring quickly (uncomment while developing)
          // ElevatedButton(
          //   onPressed: () {
          //     setState(() {
          //       _scores = [
          //         {"label": "LEAF_RUST", "prob": 0.78},
          //         {"label": "LOOPER_CATERPILLARS", "prob": 0.64},
          //         {"label": "MOSQUITO_BUG", "prob": 0.33},
          //       ];
          //       _advice = "Sample advice (debug)";
          //     });
          //   },
          //   child: Text("Load sample scores (debug)"),
          // ),

          const SizedBox(height: 12),

          if (_scores.isNotEmpty) ...[
            Wrap(
              spacing: 12,
              children: [
                for (var s in _scores.take(2))
                  Chip(label: Text("${s['label']}")),
              ],
            ),
            const SizedBox(height: 12),
          ],

          const Text('All scores:',
              style: TextStyle(fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          _buildScoreChips(),
          const SizedBox(height: 18),

          if (_advice.isNotEmpty) ...[
            const Text("Advice:",
                style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
            const SizedBox(height: 8),
            Text(_advice),
            const SizedBox(height: 12),
          ],

          if (_debug.isNotEmpty) ...[
            const Text("Debug:", style: TextStyle(fontWeight: FontWeight.bold)),
            const SizedBox(height: 6),
            Text(jsonEncode(_debug)),
            const SizedBox(height: 12),
          ],
        ]),
      ),
    );
  }
}
