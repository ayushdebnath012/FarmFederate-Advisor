import 'package:platform/platform.dart';

const _plat = LocalPlatform();

bool get isWeb => identical(0, 0.0);
bool get isMobile => _plat.isAndroid || _plat.isIOS;
bool get isDesktop => _plat.isMacOS || _plat.isWindows || _plat.isLinux;
