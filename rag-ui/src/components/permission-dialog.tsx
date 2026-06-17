import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from './ui/dialog';
import { Button } from './ui/button';

interface PermissionDialogProps {
  open: boolean;
  onClose: () => void;
  onRetry: () => void;
}

export function PermissionDialog({ open, onClose, onRetry }: PermissionDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>需要字体访问权限</DialogTitle>
          <DialogDescription>
            此应用需要读取您系统中的字体列表，以便您选择和预览。请点击上方浏览器地址栏旁的权限提示，允许访问字体，然后点击重试。
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>使用预设字体</Button>
          <Button onClick={onRetry}>重试</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}