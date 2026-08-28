import 'package:flutter/material.dart';
import '../models/sensor_message.dart';

class SensorCard extends StatelessWidget {
  final SensorMessage msg;
  const SensorCard({super.key, required this.msg});
  @override
  Widget build(BuildContext context) {
    final data = msg.data;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              msg.topic,
              style: const TextStyle(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            ...data.entries.map(
              (e) => Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(e.key),
                  Text(
                    "${e.value}",
                    style: const TextStyle(fontWeight: FontWeight.bold),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 6),
            Text(
              "at: ${msg.at.toLocal()}",
              style: const TextStyle(fontSize: 10, color: Colors.grey),
            ),
          ],
        ),
      ),
    );
  }
}
