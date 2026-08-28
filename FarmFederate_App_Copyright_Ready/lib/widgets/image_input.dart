import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:file_picker/file_picker.dart';

typedef ImageCallback = void Function(File?);

class ImageInput extends StatefulWidget {
  final ImageCallback onImage;
  const ImageInput({Key? key, required this.onImage}) : super(key: key);

  @override
  _ImageInputState createState() => _ImageInputState();
}

class _ImageInputState extends State<ImageInput> {
  File? _file;
  final ImagePicker _picker = ImagePicker();

  Future<void> _pickCamera() async {
    final XFile? photo = await _picker.pickImage(
      source: ImageSource.camera,
      imageQuality: 80,
    );
    if (photo != null) {
      setState(() {
        _file = File(photo.path);
      });
      widget.onImage(_file);
    }
  }

  Future<void> _pickFile() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.image,
      withData: false,
    );
    if (result != null && result.files.isNotEmpty) {
      final p = result.files.first.path;
      if (p != null) {
        setState(() {
          _file = File(p);
        });
        widget.onImage(_file);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        ElevatedButton.icon(
          onPressed: _pickCamera,
          icon: const Icon(Icons.camera_alt),
          label: const Text("Camera"),
        ),
        const SizedBox(width: 8),
        ElevatedButton.icon(
          onPressed: _pickFile,
          icon: const Icon(Icons.upload_file),
          label: const Text("Upload"),
        ),
        if (_file != null) ...[
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              "Selected: ${_file!.path.split('/').last}",
              overflow: TextOverflow.ellipsis,
            ),
          ),
        ],
      ],
    );
  }
}
