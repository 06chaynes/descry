import 'package:gt/io.dart';
import 'package:gt/models.dart';

void main() {
  final user = User();
  user.rename('ann');
  load(Reader(), '[1]');
}
