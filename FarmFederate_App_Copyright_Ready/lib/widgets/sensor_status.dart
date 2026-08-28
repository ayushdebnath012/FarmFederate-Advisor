import 'package:flutter/material.dart';

class SensorStatus extends StatelessWidget {
  final String? latestSensors;
  final DateTime? lastUpdated;

  const SensorStatus({Key? key, this.latestSensors, this.lastUpdated})
      : super(key: key);

  @override
  Widget build(BuildContext context) {
    final ok = latestSensors != null;
    final subtitle = ok
        ? (lastUpdated != null
            ? 'Last: ${_fmtTimeAgo(lastUpdated!)}'
            : 'Sensor data available')
        : 'No sensor data';

    return Row(
      children: [
        Icon(
          ok ? Icons.cloud_done : Icons.cloud_off,
          color: ok ? Colors.green : Colors.red,
        ),
        const SizedBox(width: 8),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                ok ? 'Sensors connected' : 'Sensors disconnected',
                style: const TextStyle(fontWeight: FontWeight.bold),
              ),
              Text(subtitle, style: const TextStyle(fontSize: 12)),
            ],
          ),
        ),
      ],
    );
  }

  String _fmtTimeAgo(DateTime dt) {
    final d = DateTime.now().difference(dt);
    if (d.inSeconds < 60) return '${d.inSeconds}s ago';
    if (d.inMinutes < 60) return '${d.inMinutes}m ago';
    if (d.inHours < 24) return '${d.inHours}h ago';
    return '${d.inDays}d ago';
  }
}
